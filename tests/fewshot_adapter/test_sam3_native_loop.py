"""测试 EfficientSAM3 原生闭环的训练集更新辅助逻辑。"""

from PIL import Image

from fewshot_adapter.data.models import Annotation, HBB, TrainingSample
from fewshot_adapter.evaluation.matching import ErrorItem
from fewshot_adapter.native.trainer import (
    _compute_round_metrics,
    _format_eval_log,
    _format_loss_log,
    _format_selected_sample_log,
    _format_trainable_module_logs,
    _render_round_visual_outputs,
    _resolve_label,
    add_selected_image_truth,
    add_selected_training_sample,
)


def test_add_selected_image_truth_adds_gt_for_false_positive_image():
    """误检图片没有 ground_truth_ids 时，也要把该图真值加入下一轮。"""
    current = [
        Annotation("seed.jpg", "seed_1", "target", "hbb", hbb=HBB(0, 0, 10, 10))
    ]
    full_gt = current + [
        Annotation("hard.jpg", "hard_1", "target", "hbb", hbb=HBB(20, 20, 30, 30)),
        Annotation("hard.jpg", "hard_other", "other", "hbb", hbb=HBB(1, 1, 2, 2)),
    ]
    selected = ErrorItem(
        image_id="hard.jpg",
        error_type="false_positive",
        risk_score=0.9,
        reason="prediction has no matching ground truth",
        ground_truth_ids=[],
        prediction_ids=["pred_1"],
        selected_for_next_round=True,
    )

    next_train = add_selected_image_truth(
        current,
        full_gt,
        selected_image_id=selected.image_id,
        label="target",
    )

    assert [annotation.object_id for annotation in next_train] == ["seed_1", "hard_1"]


def test_resolve_label_requires_explicit_label_for_multiclass_data():
    """多类别数据不能静默取第一类，避免 GPU 验证时训练错目标。"""
    annotations = [
        Annotation("a.jpg", "a_1", "car", "hbb", hbb=HBB(0, 0, 1, 1)),
        Annotation("b.jpg", "b_1", "ship", "hbb", hbb=HBB(0, 0, 1, 1)),
    ]

    try:
        _resolve_label(None, annotations)
    except ValueError as exc:
        assert "--label" in str(exc)
    else:
        raise AssertionError("expected multi-class data to require an explicit label")


def test_compute_round_metrics_filters_to_target_label():
    """每轮 summary 指标只评估当前目标类别，避免其他类别污染 recall。"""
    metrics = _compute_round_metrics(
        full_ground_truth=[
            Annotation("target.jpg", "target_1", "target", "hbb", hbb=HBB(0, 0, 10, 10)),
            Annotation("other.jpg", "other_1", "other", "hbb", hbb=HBB(0, 0, 10, 10)),
        ],
        predictions=[],
        target_label="target",
        iou_threshold=0.5,
        iou_mode="hbb",
    )

    assert metrics["ground_truth_count"] == 1
    assert metrics["prediction_count"] == 0
    assert metrics["fn"] == 1


def test_render_round_visual_outputs_returns_summary_paths(tmp_path):
    """训练器层要把本轮可视化目录写进 round summary。"""
    image_path = tmp_path / "target.jpg"
    Image.new("RGB", (48, 48), "white").save(image_path)
    annotation = Annotation("target.jpg", "target_1", "target", "hbb", hbb=HBB(5, 5, 20, 20))

    summary_paths = _render_round_visual_outputs(
        round_dir=tmp_path / "round_00",
        image_map={"target.jpg": str(image_path)},
        current_train=[annotation],
        full_ground_truth=[annotation],
        predictions=[],
        errors=[],
    )

    assert set(summary_paths) == {"train_inputs", "errors_vis", "predictions_vis"}
    assert (tmp_path / "round_00" / "train_inputs" / "target.jpg_gt.jpg").is_file()
    assert (tmp_path / "round_00" / "predictions_vis" / "target.jpg_pred.jpg").is_file()


def test_add_selected_training_sample_adds_background_false_positive_as_negative():
    """训练器应把纯背景误检加入下一轮 hard negative。"""
    current = [
        TrainingSample(
            image_id="seed.jpg",
            label="target",
            annotations=[Annotation("seed.jpg", "seed_1", "target", "hbb", hbb=HBB(0, 0, 1, 1))],
        )
    ]
    selected = ErrorItem(
        image_id="background.jpg",
        error_type="false_positive",
        risk_score=0.9,
        reason="prediction has no matching ground truth",
        ground_truth_ids=[],
        prediction_ids=["background.jpg:0000"],
        selected_for_next_round=True,
    )

    next_samples = add_selected_training_sample(
        current,
        all_ground_truths=current[0].annotations,
        selected=selected,
        label="target",
    )

    assert next_samples[-1].sample_type == "negative"
    assert next_samples[-1].image_id == "background.jpg"


def test_format_trainable_module_logs_groups_parameter_names():
    """训练启动日志要列出实际微调模块和参数量，方便 GPU 验证时确认冻结策略。"""

    class FakeParameter:
        requires_grad = True

        def __init__(self, count):
            self.count = count

        def numel(self):
            return self.count

    class FakeWrapper:
        def named_parameters(self):
            return iter(
                [
                    ("task_prompt_tokens", FakeParameter(8)),
                    ("prompt_adapter.down.weight", FakeParameter(16)),
                    ("model.dot_prod_scoring.prompt_mlp.weight", FakeParameter(32)),
                    ("model.backbone.frozen.weight", type("Frozen", (), {"requires_grad": False})()),
                ]
            )

    lines = _format_trainable_module_logs(FakeWrapper())

    assert lines[0] == "[fewshot] 本次可微调模块：3 个参数张量，共 56 个参数"
    assert any("task_prompt_tokens" in line for line in lines)
    assert any("prompt_adapter" in line for line in lines)
    assert any("dot_prod_scoring" in line for line in lines)
    assert all("frozen" not in line for line in lines)


def test_format_loss_log_includes_learning_rate_and_key_losses():
    """训练 step 日志要包含学习率和关键 loss 字段。"""
    line = _format_loss_log(
        round_index=0,
        step=9,
        steps=80,
        learning_rates=[8e-5],
        losses={
            "core_loss": 1.23456,
            "loss_ce": 0.5,
            "loss_bbox": 0.25,
            "loss_giou": 0.125,
            "presence_loss": 0.0625,
        },
    )

    assert "round=1" in line
    assert "step=10/80" in line
    assert "lr=8e-05" in line
    assert "core_loss=1.2346" in line
    assert "presence_loss=0.0625" in line


def test_format_eval_and_selected_sample_logs_are_readable():
    """每轮结束日志要清楚展示指标和下一轮选样类型。"""
    eval_line = _format_eval_log(
        round_index=1,
        prediction_count=12,
        error_count=3,
        metrics={"precision": 0.5, "recall": 0.25, "f1": 0.3333, "miou": 0.7},
    )
    selected_line = _format_selected_sample_log(
        selected=ErrorItem(
            image_id="background.jpg",
            error_type="false_positive",
            risk_score=0.9,
            reason="prediction has no matching ground truth",
            ground_truth_ids=[],
            prediction_ids=["background.jpg:0000"],
            selected_for_next_round=True,
        ),
        next_train=[
            TrainingSample(
                image_id="background.jpg",
                label="target",
                annotations=[],
                sample_type="negative",
            )
        ],
        label="target",
    )

    assert "round=2" in eval_line
    assert "pred=12" in eval_line
    assert "P=0.5000" in eval_line
    assert "mIoU=0.7000" in eval_line
    assert "no-object 负样本" in selected_line
    assert "background.jpg" in selected_line
