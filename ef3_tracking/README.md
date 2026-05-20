# ef3_tracking -- EfficientSAM3 video tracking on edge devices

A small, modular wrapper around the EfficientSAM3 video predictor that targets
edge GPUs (NVIDIA Jetson Orin AGX / Orin NX) but also runs fine on a desktop
CUDA box or CPU.

It exposes **two completely separate tracking modes**, both built on top of the
same `Sam3VideoPredictorMultiGPU` already shipped in this repo:

| Mode              | Class           | Seed prompt                | When to use |
| ----------------- | --------------- | --------------------------- | ----------- |
| Manual selection  | `ManualTracker` | Box and/or clicks on frame 0 | The user knows exactly which object to track and points at it. No language model is touched, so this is the lightest path on the edge. |
| Text grounding    | `TextTracker`   | Natural-language string     | "Track the red car." The ViT-based text encoder (MobileCLIP student by default) grounds the prompt on the seed frame; the predictor propagates each grounded instance through the video. |

Both modes share the same:

* `VideoReader` / `VideoWriter` (MP4 *or* JPEG-frame directory),
* `TrackingPipeline` that wires source -> tracker -> writer,
* annotated-frame renderer (`annotate_frame`),
* edge-device knobs (`EdgeConfig` and its `for_orin_agx()` / `for_orin_nx()` /
  `for_cpu()` presets).

## Install

The base SAM3 stack is the only hard requirement. From the repo root:

```bash
pip install -e .
pip install -e sam3            # the upstream SAM3 package
pip install opencv-python-headless numpy Pillow pytest
```

On Jetson Orin AGX use the JetPack-bundled CUDA + cuDNN; install PyTorch via
the NVIDIA-provided wheel that matches your JetPack version.

## Quick start -- text prompt

```bash
python -m ef3_tracking.cli.track_text \
    --video sample.mp4 \
    --output tracked.mp4 \
    --prompt "the red car" \
    --preset orin-agx
```

Or from Python:

```python
from ef3_tracking import EdgeConfig, TextTracker, TrackingPipeline, VideoReader, VideoWriter
from ef3_tracking.backends import build_edge_backend

cfg = EdgeConfig.for_orin_agx()
backend = build_edge_backend(cfg)

reader = VideoReader("sample.mp4")
writer = VideoWriter("tracked.mp4", width=reader.width, height=reader.height, fps=reader.metadata.fps)

tracker = TextTracker(backend)
pipeline = TrackingPipeline(tracker, reader, writer)
with pipeline.session():
    pipeline.seed_with_text("the red car")
    result = pipeline.run()

print(f"tracked {result.num_objects()} object(s) across {result.num_frames()} frames")
```

## Quick start -- manual selection

```bash
# Bounding box on the seed frame (pixel coords)
python -m ef3_tracking.cli.track_manual \
    --video sample.mp4 \
    --output tracked.mp4 \
    --box 320,240,640,540 \
    --preset orin-agx
```

```bash
# Or one or more clicks (label=1 positive, label=0 negative)
python -m ef3_tracking.cli.track_manual \
    --video sample.mp4 \
    --output tracked.mp4 \
    --point 470,395 \
    --point 600,260,0
```

Python equivalent:

```python
from ef3_tracking import EdgeConfig, ManualTracker, TrackingPipeline, VideoReader, VideoWriter
from ef3_tracking.backends import build_edge_backend
from ef3_tracking.manual_tracker import make_box_selection

cfg = EdgeConfig.for_orin_agx()
backend = build_edge_backend(cfg)

reader = VideoReader("sample.mp4")
writer = VideoWriter("tracked.mp4", width=reader.width, height=reader.height, fps=reader.metadata.fps)

tracker = ManualTracker(backend, label="car")
pipeline = TrackingPipeline(tracker, reader, writer)
with pipeline.session():
    pipeline.seed_with_selections([make_box_selection(320, 240, 640, 540)])
    result = pipeline.run()
```

## Architecture

```
ef3_tracking/
├── config.py            EdgeConfig + presets (Orin AGX / NX / CPU)
├── prompts.py           PointPrompt, BoxPrompt, ManualSelection, TextPrompt
├── video_io.py          VideoReader / VideoWriter (mp4 + frame-dir)
├── visualization.py     mask overlay, bbox drawing, palette
├── tracker.py           BaseTracker + BackendProtocol + TrackingResult
├── manual_tracker.py    ManualTracker
├── text_tracker.py      TextTracker
├── pipeline.py          TrackingPipeline
├── backends/
│   ├── mock.py          MockBackend (used by tests & demo)
│   └── sam3_backend.py  Real backend; lazily imports the SAM3 stack
├── cli/
│   ├── track_manual.py  python -m ef3_tracking.cli.track_manual ...
│   └── track_text.py    python -m ef3_tracking.cli.track_text ...
├── demo.py              Offline pipeline demo (mock backend, no weights)
└── tests/               95 pytest cases (no GPU / no model weights needed)
```

Everything talks to the predictor through the small `BackendProtocol` in
`tracker.py`. The real backend is in `backends/sam3_backend.py` and only
imports the SAM3 stack when `build_edge_backend()` is actually called, so the
package can be used (and tested) on machines without `timm`, MobileCLIP, etc.

## Edge-device notes

`EdgeConfig.for_orin_agx()` defaults to:

| Knob                 | Value           | Reason |
| -------------------- | --------------- | ------ |
| `backbone_type`      | `efficientvit`  | Cheapest vision encoder shipped in the repo. |
| `model_name`         | `b0`            | Smallest EfficientViT variant. |
| `text_encoder_type`  | `MobileCLIP-S0` | LiteText student encoder; ctx=16 for short prompts. |
| `precision`          | `fp16`          | Half precision on Ampere/Orin. |
| `frame_stride`       | `1`             | No subsampling; bump to 2 if you saturate the GPU. |
| `max_resolution`     | `720`           | 720p inputs comfortably fit on Orin AGX 32GB. |

`for_orin_nx()` tightens the same knobs (smaller resolution, frame_stride=2,
max_num_objects=4); `for_cpu()` falls back to fp32 + CPU for debugging.

Override anything you like:

```python
cfg = EdgeConfig.for_orin_agx()
cfg.frame_stride = 2
cfg.checkpoint_path = "/path/to/efficient_sam3_efficientvit_b0.pt"
cfg.load_from_hf = False
```

## Testing

The tests use a `MockBackend` that synthesises masks deterministically, so they
run in ~1.3s without GPU or model weights:

```bash
pytest ef3_tracking/tests/ -v
```

Or run the offline pipeline demo (writes annotated MP4s to `--out`):

```bash
python -m ef3_tracking.demo --out /tmp/demo_out
```
