from pathlib import Path
from types import SimpleNamespace

from fewshot_lora.config import EvaluationConfig, FewShotLoRAConfig
from fewshot_lora.eval.metrics import ImageEval
from fewshot_lora.runtime import runner
from fewshot_lora.types import TrainRoundOutput


def _image(image_name: str, has_instance: bool = True):
    return SimpleNamespace(
        image_name=image_name,
        instances=[object()] if has_instance else [],
    )


def _eval(image_id: str, failed: bool) -> ImageEval:
    return ImageEval(
        image_id=image_id,
        matches=(),
        localization_errors=(),
        false_positive_indices=(),
        false_negative_indices=(0,) if failed else (),
    )


def test_evaluate_round_reloads_adapter_into_fresh_model(monkeypatch, tmp_path: Path):
    dataset = SimpleNamespace(images=[_image("seed")], issues=[])
    build_calls = []
    loaded = []
    evaluated = []

    def fake_build_trainable_model(config):
        model = SimpleNamespace(name=f"model-{len(build_calls)}")
        build_calls.append(model)
        return model, object(), object()

    def fake_load_lora_adapter(model, adapter_path):
        loaded.append((model, adapter_path))

    def fake_evaluate_images(model, images, config):
        evaluated.append(model)
        return [_eval("seed", failed=False)]

    monkeypatch.setattr(runner, "load_detect_dataset", lambda dataset_dir, annotation_filename: dataset)
    monkeypatch.setattr(runner, "build_trainable_model", fake_build_trainable_model)
    monkeypatch.setattr(runner, "load_lora_adapter", fake_load_lora_adapter)
    monkeypatch.setattr(runner, "evaluate_images", fake_evaluate_images)
    monkeypatch.setattr(runner, "write_dataset_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "train_lora_round",
        lambda **kwargs: TrainRoundOutput(
            adapter_path=kwargs["adapter_path"],
            train_image_count=1,
            train_instance_count=1,
            train_seconds=0.1,
            train_steps=1,
        ),
    )

    config = FewShotLoRAConfig(output_dir=tmp_path, max_rounds=1)
    summary = runner.run_one_dataset(Path("dataset"), config)

    adapter_path = summary.rounds[0].adapter_path
    assert len(build_calls) == 2
    assert loaded == [(build_calls[1], adapter_path)]
    assert evaluated == [build_calls[1]]


def test_evaluate_round_can_use_in_memory_model_when_reload_disabled(monkeypatch, tmp_path: Path):
    dataset = SimpleNamespace(images=[_image("seed")], issues=[])
    build_calls = []
    loaded = []
    evaluated = []

    def fake_build_trainable_model(config):
        model = SimpleNamespace(name=f"model-{len(build_calls)}")
        build_calls.append(model)
        return model, object(), object()

    monkeypatch.setattr(runner, "load_detect_dataset", lambda dataset_dir, annotation_filename: dataset)
    monkeypatch.setattr(runner, "build_trainable_model", fake_build_trainable_model)
    monkeypatch.setattr(runner, "load_lora_adapter", lambda model, adapter_path: loaded.append((model, adapter_path)))
    monkeypatch.setattr(runner, "evaluate_images", lambda model, images, config: evaluated.append(model) or [_eval("seed", failed=False)])
    monkeypatch.setattr(runner, "write_dataset_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        runner,
        "train_lora_round",
        lambda **kwargs: TrainRoundOutput(
            adapter_path=kwargs["adapter_path"],
            train_image_count=1,
            train_instance_count=1,
            train_seconds=0.1,
            train_steps=1,
        ),
    )

    config = FewShotLoRAConfig(
        output_dir=tmp_path,
        max_rounds=1,
        evaluation=EvaluationConfig(reload_adapter_for_eval=False),
    )
    runner.run_one_dataset(Path("dataset"), config)

    assert len(build_calls) == 1
    assert loaded == []
    assert evaluated == [build_calls[0]]


def test_next_round_training_reuses_same_model_with_previous_lora_state(monkeypatch, tmp_path: Path):
    dataset = SimpleNamespace(images=[_image("seed"), _image("bad")], issues=[])
    model = SimpleNamespace(lora_marker="initial")
    observed_model_ids = []
    observed_markers = []

    monkeypatch.setattr(runner, "load_detect_dataset", lambda dataset_dir, annotation_filename: dataset)
    monkeypatch.setattr(runner, "build_trainable_model", lambda config: (model, object(), object()))
    monkeypatch.setattr(runner, "write_dataset_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "evaluate_images", lambda model_arg, images, config: [[_eval("bad", True)], [_eval("bad", False)]][len(observed_model_ids) - 1])

    def fake_train_lora_round(**kwargs):
        observed_model_ids.append(id(kwargs["model"]))
        observed_markers.append(kwargs["model"].lora_marker)
        kwargs["model"].lora_marker = f"round-{kwargs['round_index']}"
        return TrainRoundOutput(
            adapter_path=kwargs["adapter_path"],
            train_image_count=len(kwargs["train_image_ids"]),
            train_instance_count=1,
            train_seconds=0.1,
            train_steps=1,
        )

    monkeypatch.setattr(runner, "train_lora_round", fake_train_lora_round)

    config = FewShotLoRAConfig(
        output_dir=tmp_path,
        max_rounds=2,
        evaluation=EvaluationConfig(reload_adapter_for_eval=False),
    )
    summary = runner.run_one_dataset(Path("dataset"), config)

    assert summary.rounds[0].continue_from_previous_round is True
    assert summary.rounds[1].continue_from_previous_round is True
    assert observed_model_ids == [id(model), id(model)]
    assert observed_markers == ["initial", "round-0"]
