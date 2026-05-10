from types import SimpleNamespace

from fewshot_lora.training import _compute_find_losses


def test_compute_find_losses_converts_batched_find_target_before_loss():
    converted = {"boxes_xyxy": "converted"}
    raw_target = object()
    batch = SimpleNamespace(find_targets=[raw_target])
    model = SimpleNamespace(back_convert=lambda target: converted)
    received = {}

    def loss_fn(outputs, targets):
        received["outputs"] = outputs
        received["targets"] = targets
        return {"core_loss": 1.0}

    outputs = object()

    losses = _compute_find_losses(model, loss_fn, outputs, batch)

    assert losses == {"core_loss": 1.0}
    assert received["outputs"] is outputs
    assert received["targets"] == [converted]
