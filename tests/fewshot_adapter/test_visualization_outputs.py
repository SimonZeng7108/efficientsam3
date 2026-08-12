"""测试每轮训练可视化图片输出。"""

from PIL import Image

from fewshot_adapter.data.models import HBB, Annotation, Prediction, TrainingSample
from fewshot_adapter.evaluation.matching import ErrorItem
from fewshot_adapter.visualization.round_outputs import render_round_visualizations


def test_render_round_visualizations_writes_three_review_folders(tmp_path):
    """每轮目录下要同时输出训练输入图、错误复查图和全量检测结果图。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    target_image = image_dir / "target.jpg"
    background_image = image_dir / "background.jpg"
    Image.new("RGB", (80, 60), "white").save(target_image)
    Image.new("RGB", (80, 60), "white").save(background_image)

    image_map = {
        "target.jpg": str(target_image),
        "background.jpg": str(background_image),
    }
    train_annotations = [
        Annotation(
            image_id="target.jpg",
            object_id="target_0001",
            label="obj",
            source_type="polygon",
            polygon=[(10, 10), (50, 8), (52, 40), (12, 42)],
        )
    ]
    full_ground_truth = train_annotations
    predictions = [
        Prediction(
            image_id="target.jpg",
            prediction_id="target.jpg:0000",
            label="obj",
            score=0.82,
            hbb=HBB(14, 12, 54, 38),
        )
    ]
    errors = [
        ErrorItem(
            image_id="target.jpg",
            error_type="localization_error",
            risk_score=0.4,
            reason="box is close but not enough",
            ground_truth_ids=["target_0001"],
            prediction_ids=["target.jpg:0000"],
            selected_for_next_round=True,
        )
    ]
    round_dir = tmp_path / "runs" / "round_00"

    outputs = render_round_visualizations(
        round_dir=round_dir,
        image_map=image_map,
        train_annotations=train_annotations,
        full_ground_truth=full_ground_truth,
        predictions=predictions,
        errors=errors,
    )

    assert outputs.train_inputs_dir == round_dir / "train_inputs"
    assert outputs.errors_dir == round_dir / "errors_vis"
    assert outputs.predictions_dir == round_dir / "predictions_vis"
    assert (round_dir / "train_inputs" / "target.jpg_gt.jpg").is_file()
    assert (round_dir / "errors_vis" / "target.jpg_error.jpg").is_file()
    assert (round_dir / "predictions_vis" / "target.jpg_pred.jpg").is_file()
    assert (round_dir / "predictions_vis" / "background.jpg_pred.jpg").is_file()


def test_prediction_visualization_keeps_background_image_without_predictions(tmp_path):
    """没有检测框的背景图也要输出原图，方便确认全量检测覆盖了该图片。"""
    image_path = tmp_path / "background.jpg"
    Image.new("RGB", (32, 24), "white").save(image_path)
    round_dir = tmp_path / "round_00"

    render_round_visualizations(
        round_dir=round_dir,
        image_map={"background.jpg": str(image_path)},
        train_annotations=[],
        full_ground_truth=[],
        predictions=[],
        errors=[],
    )

    rendered = Image.open(round_dir / "predictions_vis" / "background.jpg_pred.jpg")
    assert rendered.size == (32, 24)


def test_train_input_visualization_writes_negative_sample_without_box(tmp_path):
    """负样本训练图要保存原图并标注 no-object，不能画成真值框。"""
    image_path = tmp_path / "background.jpg"
    Image.new("RGB", (40, 30), "white").save(image_path)
    round_dir = tmp_path / "round_00"

    render_round_visualizations(
        round_dir=round_dir,
        image_map={"background.jpg": str(image_path)},
        train_annotations=[],
        training_samples=[
            TrainingSample(
                image_id="background.jpg",
                label="obj",
                annotations=[],
                sample_type="negative",
                reason="false_positive",
            )
        ],
        full_ground_truth=[],
        predictions=[],
        errors=[],
    )

    assert (round_dir / "train_inputs" / "background.jpg_gt.jpg").is_file()
