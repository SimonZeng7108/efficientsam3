"""Evaluate student text encoder quality via cosine similarity vs SAM3 teacher.

Text-only evaluation:
- Teacher: SAM3 language backbone (no vision encoder, no segmentation head).
- Student: EfficientSAM3 stage1 text student encoder (MobileCLIP variants).

We run both on noun-phrase annotations and compare token-level features.

Input JSON (NP-only):
    data/sa-v-text/sa-co-veval/saco_veval_noun_phrases.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def _ensure_import_paths() -> None:
    # Use repo root on sys.path so imports like `sam3.sam3.*` work.
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


def _load_np_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "noun_phrases" in data:
        phrases = data["noun_phrases"]
        if not isinstance(phrases, list) or (phrases and not isinstance(phrases[0], str)):
            raise ValueError(f"{path}: expected 'noun_phrases' to be a list[str]")
        return phrases
    raise ValueError(f"{path}: expected NP-only JSON with key 'noun_phrases'")


def _load_student_checkpoint(path: str) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"{path}: expected checkpoint dict")
    state_dict = ckpt.get("model") if isinstance(ckpt.get("model"), dict) else ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path}: expected state_dict under 'model' or at top-level")

    clean_state: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        clean_state[k] = v
    return ckpt, clean_state


def _build_student_text_only(config) -> torch.nn.Module:
    # Mirrors stage1/model.py build_text_student_model but without importing
    # any vision backbones or other stage1 training utilities.
    from sam3.model.text_encoder_student import TextStudentEncoder

    backbone = str(config.MODEL.BACKBONE)

    cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "dim": 512,
        "ffn_multiplier_per_layer": 4.0,
        "n_heads_per_layer": 8,
        "n_transformer_layers": 12,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "model_name": "base",
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    }

    if backbone == "MobileCLIP-S0":
        cfg.update(
            {
                "dim": 512,
                "n_transformer_layers": 4,
                "n_heads_per_layer": 8,
                "model_name": "mct",
                "ffn_multiplier_per_layer": 4.0,
            }
        )
    elif backbone in ["MobileCLIP-S1", "MobileCLIP2-S0", "MobileCLIP2-S2"]:
        cfg.update(
            {
                "dim": 512,
                "n_transformer_layers": 12,
                "n_heads_per_layer": 8,
                "model_name": "base",
            }
        )
    elif backbone == "MobileCLIP-B":
        cfg.update(
            {
                "dim": 512,
                "n_transformer_layers": 12,
                "n_heads_per_layer": 8,
                "model_name": "base",
                "causal_masking": True,
            }
        )
    elif backbone in ["MobileCLIP2-S3", "MobileCLIP2-S4", "MobileCLIP2-L"]:
        cfg.update(
            {
                "dim": 768,
                "n_transformer_layers": 12,
                "n_heads_per_layer": 12,
                "model_name": "base",
            }
        )

    # In stage1 text distillation, student context_length is matched to teacher.
    return TextStudentEncoder(
        cfg=cfg,
        context_length=32,
        output_dim=int(config.DISTILL.EMBED_DIM),
    )


def _build_teacher_text_only(teacher_ckpt: str | None, device: torch.device) -> torch.nn.Module:
    # Build SAM3 with ONLY the text encoder path.
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=teacher_ckpt,
        load_from_HF=True if teacher_ckpt is None else False,
        eval_mode=True,
        device=str(device),
        enable_segmentation=False,
        enable_inst_interactivity=False,
        compile=False,
        enable_text_encoder=True,
        enable_vision_encoder=False,
    )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    student_ckpts: List[str],
    np_json: str,
    device: str,
    batch_size: int,
    teacher_ckpt: str | None,
    max_texts: int | None,
) -> None:
    phrases = _load_np_json(np_json)
    if max_texts is not None:
        phrases = phrases[:max_texts]
    print(f"Loaded {len(phrases)} noun phrases from {np_json}")

    dev = torch.device(device)

    print("Building SAM3 teacher (text-only)...")
    teacher_sam3 = _build_teacher_text_only(teacher_ckpt=teacher_ckpt, device=dev)

    results: List[Tuple[str, float, int]] = []

    for student_ckpt in student_ckpts:
        ckpt, student_state = _load_student_checkpoint(student_ckpt)
        config = ckpt.get("config")
        if config is None:
            raise ValueError(
                f"{student_ckpt}: missing 'config' in checkpoint. "
                "This script expects stage1 standalone text checkpoints with embedded config."
            )

        print(f"\nEvaluating student checkpoint: {student_ckpt}")
        print(f"Student backbone: {config.MODEL.BACKBONE}")

        student = _build_student_text_only(config).to(dev)
        student.eval()

        missing, unexpected = student.load_state_dict(student_state, strict=False)
        if missing or unexpected:
            print(f"Student load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

        context_len = int(getattr(student, "context_length", 32))

        total_sim = 0.0
        total_tokens = 0.0

        pbar = tqdm(range(0, len(phrases), batch_size), desc=f"eval:{Path(student_ckpt).name}")
        for start in pbar:
            batch = phrases[start : start + batch_size]

            # Teacher language backbone returns (mask, memory, embeds)
            t_mask, t_mem, _ = teacher_sam3.backbone.language_backbone(
                batch, input_boxes=None, device=dev
            )
            # Student returns (mask, memory, embeds)
            s_mask, s_mem, _ = student(batch, device=dev)

            if t_mem.shape != s_mem.shape:
                raise RuntimeError(
                    f"Feature shape mismatch: teacher={tuple(t_mem.shape)} student={tuple(s_mem.shape)}"
                )

            tokenized = student.tokenizer(batch, context_length=context_len).to(dev)
            valid = (tokenized != 0).float()  # [B, S]

            # memories are [Seq, B, C] -> [B, S, C]
            t = F.normalize(t_mem.transpose(0, 1), dim=-1)
            s = F.normalize(s_mem.transpose(0, 1), dim=-1)
            sim = (t * s).sum(dim=-1)  # [B, S]

            sim_sum = float((sim * valid).sum().item())
            tok_sum = float(valid.sum().item())
            total_sim += sim_sum
            total_tokens += tok_sum

            if total_tokens > 0:
                pbar.set_postfix(avg_cos=total_sim / total_tokens)

        avg = total_sim / max(total_tokens, 1.0)
        results.append((student_ckpt, avg, int(total_tokens)))

        print("RESULT")
        print(f"student_ckpt: {student_ckpt}")
        print(f"texts: {len(phrases)}")
        print(f"total_valid_tokens: {int(total_tokens)}")
        print(f"avg_token_cosine_similarity: {avg:.6f}")

    if len(results) > 1:
        print("\nSUMMARY")
        for ckpt_path, avg, tok in results:
            print(f"{ckpt_path}\tavg_cos={avg:.6f}\ttokens={tok}")


def main() -> None:
    _ensure_import_paths()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student-ckpt",
        required=True,
        nargs="+",
        help="One or more stage1 standalone text checkpoints (.pth)",
    )
    parser.add_argument(
        "--np-json",
        required=True,
        help="Path to NP-only JSON (with key 'noun_phrases')",
    )
    parser.add_argument(
        "--teacher-ckpt",
        default=None,
        help="Optional SAM3 checkpoint path to use for the teacher (overrides checkpoint config.MODEL.RESUME)",
    )
    from sam3.device import get_device
    parser.add_argument("--device", default=str(get_device()))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="Optional cap for quick runs",
    )
    args = parser.parse_args()

    evaluate(
        student_ckpts=args.student_ckpt,
        np_json=args.np_json,
        device=args.device,
        batch_size=args.batch_size,
        teacher_ckpt=args.teacher_ckpt,
        max_texts=args.max_texts,
    )


if __name__ == "__main__":
    main()
