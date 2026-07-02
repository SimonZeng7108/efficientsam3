"""测试第 0 轮初始训练集选择逻辑。"""

from fewshot_adapter.data.models import HBB, Annotation
from fewshot_adapter.data.sampling import create_initial_train_set


def test_create_initial_train_set_selects_one_labeled_image_with_seed():
    annotations = [
        Annotation("img_a", "gt_a", "target", "hbb", hbb=HBB(0, 0, 1, 1)),
        Annotation("img_b", "gt_b", "target", "hbb", hbb=HBB(1, 1, 2, 2)),
        Annotation("img_c", "gt_c", "other", "hbb", hbb=HBB(2, 2, 3, 3)),
    ]

    selected = create_initial_train_set(annotations, label="target", seed=0)

    assert len(selected) == 1
    assert selected[0].label == "target"
    assert selected[0].image_id in {"img_a", "img_b"}


def test_create_initial_train_set_keeps_all_target_annotations_on_selected_image():
    annotations = [
        Annotation("img_a", "gt_a1", "target", "hbb", hbb=HBB(0, 0, 1, 1)),
        Annotation("img_a", "gt_a2", "target", "hbb", hbb=HBB(1, 1, 2, 2)),
        Annotation("img_a", "gt_other", "other", "hbb", hbb=HBB(2, 2, 3, 3)),
    ]

    selected = create_initial_train_set(annotations, label="target", seed=0)

    assert [annotation.object_id for annotation in selected] == ["gt_a1", "gt_a2"]
