"""测试 EfficientSAM3 原生少样本闭环的配置安全检查。"""

import pytest

from fewshot_adapter.data.models import Annotation, HBB, TrainingSample
from fewshot_adapter.native.loss import NativeLossConfig
from fewshot_adapter.native import trainer as trainer_module
from fewshot_adapter.native.trainer import (
    NativeFewShotLoopConfig,
    run_native_fewshot_loop,
    train_native_adapter_one_round,
)


def test_native_loop_requires_segmentation_head_when_mask_loss_enabled(tmp_path):
    """mask loss 需要 SAM3 segmentation head，否则会在深层 loss 处缺少 pred_masks。"""
    config = NativeFewShotLoopConfig(
        checkpoint="ckpt.pt",
        output_root=str(tmp_path / "runs"),
        enable_segmentation=False,
    )

    with pytest.raises(ValueError, match="ENABLE_SEGMENTATION"):
        run_native_fewshot_loop(
            full_ground_truth_path=tmp_path / "missing_gt.json",
            image_map_path=tmp_path / "missing_image_map.json",
            config=config,
            loss_config=NativeLossConfig(use_masks=True),
            log_fn=None,
        )


def test_train_one_round_passes_mask_flag_to_batch_builder(monkeypatch):
    """单轮训练开启 mask loss 时，构造 SAM3 target 必须带上 include_masks=True。"""
    captured_include_masks = []

    def fake_build_sam3_training_batch_from_samples(samples, image_map, *, resolution, device, include_masks):
        captured_include_masks.append(include_masks)
        return _FakeBatch()

    monkeypatch.setattr(
        trainer_module,
        "build_sam3_training_batch_from_samples",
        fake_build_sam3_training_batch_from_samples,
    )
    monkeypatch.setattr(trainer_module, "require_torch", lambda: _FakeTorch())

    sample = TrainingSample(
        image_id="img.jpg",
        label="obj",
        annotations=[Annotation("img.jpg", "gt_1", "obj", "hbb", hbb=HBB(0, 0, 1, 1))],
    )

    history = train_native_adapter_one_round(
        wrapper=_FakeWrapper(),
        optimizer=_FakeOptimizer(),
        loss_fn=_FakeLossFn(),
        training_samples=[sample],
        image_map={"img.jpg": "img.jpg"},
        steps=1,
        resolution=8,
        device="cuda",
        sam3_output_cls=_FakeSam3Output,
        include_masks=True,
    )

    assert captured_include_masks == [True]
    assert history[0]["loss_mask"] == pytest.approx(2.0)


class _FakeBatch:
    find_target = object()


class _FakeModel:
    def back_convert(self, find_target):
        return {"target": find_target}


class _FakeWrapper:
    model = _FakeModel()

    def train(self):
        self.training = True

    def forward_batch(self, batch):
        return {"pred": batch}


class _FakeOptimizer:
    def zero_grad(self, *, set_to_none):
        self.zeroed = set_to_none

    def step(self):
        self.stepped = True


class _FakeScalar:
    def __init__(self, value):
        self.value = float(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def backward(self):
        self.backward_called = True

    def __float__(self):
        return self.value


class _FakeFinite:
    def all(self):
        return True


class _FakeTorch:
    def isfinite(self, value):
        return _FakeFinite()


class _FakeLossFn:
    def __call__(self, sam3_output, targets):
        return {
            "core_loss": _FakeScalar(1.0),
            "loss_mask": _FakeScalar(2.0),
        }


class _FakeSam3Output:
    def __init__(self, payload):
        self.payload = payload
