from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class RunningMetric:
    def __init__(self, bin_edges: np.ndarray):
        self.bin_edges = bin_edges
        self.nbins = len(bin_edges) - 1
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.hist = np.zeros(self.nbins, dtype=np.int64)
        self.out_of_range = 0

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            return

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min = value if value < self.min else self.min
        self.max = value if value > self.max else self.max

        if value < 0.0 or value > 1.0:
            self.out_of_range += 1

        index = int(np.searchsorted(self.bin_edges, value, side="right") - 1)
        index = max(0, min(self.nbins - 1, index))
        self.hist[index] += 1

    def summary(self) -> Dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "out_of_range": int(self.out_of_range),
            }

        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "std": float(math.sqrt(variance)),
            "min": float(self.min),
            "max": float(self.max),
            "out_of_range": int(self.out_of_range),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute predicted_iou / stability_score stats for SA-1B masks."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/sa-1b-1p_reorg",
        help="Dataset root containing annotations/{train,val}/*.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/stage3/sa1b_1p_reorg_quality_stats",
        help="Directory for stats report and diagrams.",
    )
    parser.add_argument("--bins", type=int, default=50, help="Number of bins in [0, 1].")
    return parser.parse_args()


def _new_metric_bucket(metrics: Iterable[str], bin_edges: np.ndarray) -> Dict[str, RunningMetric]:
    return {name: RunningMetric(bin_edges=bin_edges) for name in metrics}


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    annotation_root = dataset_root / "annotations"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ("predicted_iou", "stability_score")
    bin_edges = np.linspace(0.0, 1.0, args.bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bar_width = (bin_edges[1] - bin_edges[0]) * 0.92

    ann_files = sorted(annotation_root.glob("*/*.json"))
    if not ann_files:
        raise RuntimeError(f"No annotation files found under: {annotation_root}")

    stats = {"all": _new_metric_bucket(metrics=metrics, bin_edges=bin_edges)}
    files_seen = {"all": 0}
    masks_seen = {"all": 0}

    for index, ann_path in enumerate(ann_files, start=1):
        split = ann_path.parent.name
        if split not in stats:
            stats[split] = _new_metric_bucket(metrics=metrics, bin_edges=bin_edges)
            files_seen[split] = 0
            masks_seen[split] = 0

        with ann_path.open("r") as fopen:
            ann_data = json.load(fopen)

        annotations = ann_data.get("annotations", [])
        files_seen["all"] += 1
        files_seen[split] += 1
        masks_seen["all"] += len(annotations)
        masks_seen[split] += len(annotations)

        for annotation in annotations:
            for metric in metrics:
                try:
                    value = float(annotation.get(metric, float("nan")))
                except (TypeError, ValueError):
                    continue

                stats["all"][metric].update(value)
                stats[split][metric].update(value)

        if index % 500 == 0:
            print(f"processed_files={index}/{len(ann_files)}")

    report = {
        "dataset_root": str(dataset_root),
        "annotation_root": str(annotation_root),
        "total_annotation_files": len(ann_files),
        "total_masks": int(masks_seen["all"]),
        "bins": {
            "count": args.bins,
            "edges": bin_edges.tolist(),
        },
        "splits": {},
    }

    for split, metric_map in stats.items():
        report["splits"][split] = {
            "annotation_files": int(files_seen.get(split, 0)),
            "mask_count": int(masks_seen.get(split, 0)),
            "metrics": {},
        }
        for metric, agg in metric_map.items():
            report["splits"][split]["metrics"][metric] = {
                **agg.summary(),
                "hist_counts": agg.hist.tolist(),
            }

    report_path = output_dir / "predicted_iou_stability_score_stats.json"
    with report_path.open("w") as fopen:
        json.dump(report, fopen, indent=2)

    # Whole-subset bar diagrams.
    for metric in metrics:
        aggregate = stats["all"][metric]
        fig, axis = plt.subplots(figsize=(12, 5))
        axis.bar(
            bin_centers,
            aggregate.hist,
            width=bar_width,
            color="#2a9d8f",
            edgecolor="#1f6f66",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel(metric)
        axis.set_ylabel("Mask count")
        axis.set_title(f"{metric} distribution across all masks ({dataset_root.name})")
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}_bar.png", dpi=180)
        plt.close(fig)

    # Extra bar diagram: split-level mean comparison.
    split_names = [name for name in stats.keys() if name != "all"]
    if split_names:
        mean_pred = [
            stats[name]["predicted_iou"].mean if stats[name]["predicted_iou"].count else 0.0
            for name in split_names
        ]
        mean_stability = [
            stats[name]["stability_score"].mean if stats[name]["stability_score"].count else 0.0
            for name in split_names
        ]

        x = np.arange(len(split_names))
        width = 0.36
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.bar(x - width / 2, mean_pred, width=width, label="predicted_iou", color="#457b9d")
        axis.bar(
            x + width / 2,
            mean_stability,
            width=width,
            label="stability_score",
            color="#e76f51",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(split_names)
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Mean score")
        axis.set_title("Mean mask quality score by split")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "mean_scores_by_split_bar.png", dpi=180)
        plt.close(fig)

    print(f"REPORT_PATH={report_path}")
    print(f"PLOT_DIR={output_dir}")
    print(f"TOTAL_FILES={len(ann_files)}")
    print(f"TOTAL_MASKS={masks_seen['all']}")
    print(f"ALL_PRED_IOU_MEAN={stats['all']['predicted_iou'].mean}")
    print(f"ALL_STABILITY_MEAN={stats['all']['stability_score'].mean}")


if __name__ == "__main__":
    main()