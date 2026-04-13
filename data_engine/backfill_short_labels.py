from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from data_engine.annotations import normalize_label
except ModuleNotFoundError:
    from annotations import normalize_label

_TEXT_REWRITE_MODEL = None
_TEXT_REWRITE_TOKENIZER = None
_TEXT_REWRITE_MODEL_NAME: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill label_5 and label_2 in existing *_enhanced.json files "
            "using a text rewrite model."
        )
    )
    parser.add_argument("--sa1b-root", default="data/SA-1B-5P", type=str)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument(
        "--text-rewrite-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        type=str,
    )
    parser.add_argument("--rewrite-max-tokens", default=96, type=int)
    parser.add_argument("--rewrite-batch-size", default=64, type=int)
    parser.add_argument("--device-map", default="auto", type=str)
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument("--max-files", default=None, type=int)
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Rewrite existing non-empty label_5 and label_2 values too.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve_torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return getattr(torch, dtype_name)


def _ensure_local_text_rewrite_model(
    model_name: str,
    device_map: str,
    dtype_name: str,
):
    global _TEXT_REWRITE_MODEL, _TEXT_REWRITE_TOKENIZER, _TEXT_REWRITE_MODEL_NAME

    if (
        _TEXT_REWRITE_MODEL is not None
        and _TEXT_REWRITE_TOKENIZER is not None
        and _TEXT_REWRITE_MODEL_NAME == model_name
    ):
        return _TEXT_REWRITE_MODEL, _TEXT_REWRITE_TOKENIZER

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _TEXT_REWRITE_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_resolve_torch_dtype(dtype_name),
        device_map=device_map,
        trust_remote_code=True,
    )
    _TEXT_REWRITE_TOKENIZER = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    # Decoder-only models should left-pad for batched generation.
    if hasattr(_TEXT_REWRITE_TOKENIZER, "padding_side"):
        _TEXT_REWRITE_TOKENIZER.padding_side = "left"
    _TEXT_REWRITE_MODEL_NAME = model_name
    return _TEXT_REWRITE_MODEL, _TEXT_REWRITE_TOKENIZER


def _build_text_rewrite_messages(
    label: str,
    max_words: int,
    exact_words: Optional[int] = None,
) -> List[Dict[str, str]]:
    limit_instruction = (
        f"Use exactly {exact_words} words."
        if exact_words is not None
        else f"Use at most {max_words} words."
    )
    system_prompt = (
        "You are an expert linguist that shortens object descriptions into concise, "
        "grammatically correct noun phrases. Return only the shortened noun phrase. "
        "Do not include punctuation, quotes, or explanations."
    )
    user_prompt = (
        f"Original label: {label}\\n\\n"
        "Shorten the label into a grammatically correct noun phrase describing the main object. "
        "Keep the most important modifiers and the core noun. "
        f"{limit_instruction}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_plain_label_line(text: Optional[str]) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0].strip("-•* \t\"'")
    if ":" in first:
        left, right = first.split(":", 1)
        if len(left.split()) <= 3 and right.strip():
            first = right.strip()
    return first


def _parse_text_rewrite_phrase(
    raw_text: Optional[str],
    max_words: int,
    exact_words: Optional[int] = None,
) -> str:
    if not raw_text:
        return ""
    phrase = normalize_label(_extract_plain_label_line(raw_text), max_words=max_words)
    if not phrase:
        return ""
    words = phrase.split()
    if exact_words is not None:
        phrase = " ".join(words[:exact_words])
    return phrase


def _chunked(items: List[Tuple[int, str]], chunk_size: int) -> Iterable[List[Tuple[int, str]]]:
    chunk_size = max(int(chunk_size), 1)
    for offset in range(0, len(items), chunk_size):
        yield items[offset : offset + chunk_size]


def _run_local_text_rewrite_batch_requests(
    labels: List[str],
    model_name: str,
    max_tokens: int,
    device_map: str,
    dtype_name: str,
    max_words: int,
    exact_words: Optional[int] = None,
) -> List[Optional[str]]:
    if not labels:
        return []

    model, tokenizer = _ensure_local_text_rewrite_model(
        model_name=model_name,
        device_map=device_map,
        dtype_name=dtype_name,
    )

    try:
        prompt_texts: List[str] = []
        for label in labels:
            messages = _build_text_rewrite_messages(
                label=label,
                max_words=max_words,
                exact_words=exact_words,
            )
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt_text = (
                    f"System: {messages[0]['content']}\\n"
                    f"User: {messages[1]['content']}\\n"
                    "Assistant:"
                )
            if isinstance(prompt_text, list):
                prompt_text = prompt_text[0]
            prompt_texts.append(str(prompt_text))

        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return [None] * len(labels)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not output_text:
            return [None] * len(labels)
        if len(output_text) < len(labels):
            output_text.extend([""] * (len(labels) - len(output_text)))
        return [str(value) for value in output_text[: len(labels)]]
    except Exception as error:
        print(f"Local text rewrite request failed: {error}")
        return [None] * len(labels)


def _assign_rewrites(
    annotations: List[Dict[str, object]],
    requests: List[Tuple[int, str]],
    key: str,
    model_name: str,
    max_tokens: int,
    batch_size: int,
    device_map: str,
    dtype_name: str,
    max_words: int,
    exact_words: Optional[int],
) -> Tuple[int, int, bool]:
    failures = 0
    rewrites = 0
    changed = False

    for batch in _chunked(requests, batch_size):
        labels = [label for _, label in batch]
        outputs = _run_local_text_rewrite_batch_requests(
            labels=labels,
            model_name=model_name,
            max_tokens=max_tokens,
            device_map=device_map,
            dtype_name=dtype_name,
            max_words=max_words,
            exact_words=exact_words,
        )
        for (annotation_index, label_10), rewrite_raw in zip(batch, outputs):
            parsed = _parse_text_rewrite_phrase(
                raw_text=rewrite_raw,
                max_words=max_words,
                exact_words=exact_words,
            )
            if not parsed:
                failures += 1
                parsed = normalize_label(label_10, max_words=max_words)
                if exact_words is not None:
                    parsed = " ".join(parsed.split()[:exact_words])

            annotation = annotations[annotation_index]
            old_value = str(annotation.get(key) or "").strip()
            if old_value != parsed:
                annotation[key] = parsed
                changed = True
            rewrites += 1

    return rewrites, failures, changed


def _annotation_files(sa1b_root: Path, split: str) -> List[Path]:
    ann_dir = sa1b_root / "annotations" / split
    return sorted(ann_dir.glob("*_enhanced.json"))


def main() -> None:
    args = parse_args()
    sa1b_root = Path(args.sa1b_root)
    files = _annotation_files(sa1b_root=sa1b_root, split=args.split)
    if args.max_files is not None:
        files = files[: args.max_files]

    files_changed = 0
    files_seen = 0
    annotations_seen = 0
    rewrite_5_requests = 0
    rewrite_2_requests = 0
    rewrite_5_failures = 0
    rewrite_2_failures = 0

    for path in files:
        files_seen += 1
        with path.open("r") as f:
            payload = json.load(f)

        annotations = payload.get("annotations", [])
        if not isinstance(annotations, list):
            continue

        annotations_seen += len(annotations)
        changed = False

        requests_5: List[Tuple[int, str]] = []
        requests_2: List[Tuple[int, str]] = []
        for idx, annotation in enumerate(annotations):
            if not isinstance(annotation, dict):
                continue
            label_10 = str(
                annotation.get("label_10")
                or annotation.get("label")
                or annotation.get("annotation_text")
                or ""
            ).strip()
            if not label_10:
                continue

            current_5 = str(annotation.get("label_5") or "").strip()
            current_2 = str(annotation.get("label_2") or "").strip()
            if args.overwrite_existing or not current_5:
                requests_5.append((idx, label_10))
            if args.overwrite_existing or not current_2:
                requests_2.append((idx, label_10))

        rewrite_5_requests += len(requests_5)
        rewrite_2_requests += len(requests_2)

        if requests_5:
            rewrites, failures, changed_5 = _assign_rewrites(
                annotations=annotations,
                requests=requests_5,
                key="label_5",
                model_name=args.text_rewrite_model,
                max_tokens=args.rewrite_max_tokens,
                batch_size=args.rewrite_batch_size,
                device_map=args.device_map,
                dtype_name=args.torch_dtype,
                max_words=5,
                exact_words=None,
            )
            rewrite_5_failures += failures
            changed = changed or changed_5
            if rewrites and (files_seen % 200 == 0):
                print(f"progress files={files_seen} rewrites_5={rewrite_5_requests}")

        if requests_2:
            rewrites, failures, changed_2 = _assign_rewrites(
                annotations=annotations,
                requests=requests_2,
                key="label_2",
                model_name=args.text_rewrite_model,
                max_tokens=args.rewrite_max_tokens,
                batch_size=args.rewrite_batch_size,
                device_map=args.device_map,
                dtype_name=args.torch_dtype,
                max_words=2,
                exact_words=2,
            )
            rewrite_2_failures += failures
            changed = changed or changed_2

        # Guarantee both keys exist for every annotation that has label_10.
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            label_10 = str(
                annotation.get("label_10")
                or annotation.get("label")
                or annotation.get("annotation_text")
                or ""
            ).strip()
            if not label_10:
                continue

            label_5 = str(annotation.get("label_5") or "").strip()
            if not label_5:
                fallback_5 = normalize_label(label_10, max_words=5)
                if str(annotation.get("label_5") or "").strip() != fallback_5:
                    annotation["label_5"] = fallback_5
                    changed = True

            label_2 = str(annotation.get("label_2") or "").strip()
            if not label_2:
                base_for_2 = str(annotation.get("label_5") or "").strip() or label_10
                fallback_2 = normalize_label(base_for_2, max_words=2)
                fallback_2 = " ".join(fallback_2.split()[:2])
                if str(annotation.get("label_2") or "").strip() != fallback_2:
                    annotation["label_2"] = fallback_2
                    changed = True

        if changed and not args.dry_run:
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            with tmp_path.open("w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")
            tmp_path.replace(path)
            files_changed += 1

    print(
        json.dumps(
            {
                "sa1b_root": str(sa1b_root),
                "split": args.split,
                "files_seen": files_seen,
                "files_changed": files_changed,
                "annotations_seen": annotations_seen,
                "rewrite_model": args.text_rewrite_model,
                "rewrite_5_requests": rewrite_5_requests,
                "rewrite_5_failures": rewrite_5_failures,
                "rewrite_2_requests": rewrite_2_requests,
                "rewrite_2_failures": rewrite_2_failures,
                "dry_run": args.dry_run,
                "overwrite_existing": args.overwrite_existing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
