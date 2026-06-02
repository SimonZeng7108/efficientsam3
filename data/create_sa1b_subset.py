#!/usr/bin/env python
import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a deterministic SA-1B image/json subset."
    )
    parser.add_argument("--source", required=True, help="SA-1B root with images/ and annotations/.")
    parser.add_argument("--output", required=True, help="Output subset root.")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="hardlink",
        help="How to materialize selected files in the subset.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path. Defaults to <output>/subset_manifest.json.",
    )
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    image_dir = source / "images" / args.split
    annotation_dir = source / "annotations" / args.split

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Missing annotation directory: {annotation_dir}")

    records = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        key = image_path.stem
        annotation_path = annotation_dir / f"{key}.json"
        if annotation_path.exists():
            records.append((key, image_path, annotation_path))

    if not records:
        raise RuntimeError(f"No image/annotation pairs found under {source}")

    num_samples = min(args.num_samples, len(records))
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(records, num_samples), key=lambda item: item[0])

    for key, image_path, annotation_path in selected:
        link_or_copy(image_path, output / "images" / args.split / image_path.name, args.mode)
        link_or_copy(
            annotation_path,
            output / "annotations" / args.split / annotation_path.name,
            args.mode,
        )

    # Keep val directories present because the repo expects this SA-1B layout.
    (output / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output / "annotations" / "val").mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else output / "subset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(source),
        "output": str(output),
        "split": args.split,
        "seed": args.seed,
        "requested_num_samples": args.num_samples,
        "actual_num_samples": len(selected),
        "mode": args.mode,
        "keys": [key for key, _, _ in selected],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Created SA-1B subset at {output} with {len(selected)} "
        f"{args.split} image/annotation pairs."
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
