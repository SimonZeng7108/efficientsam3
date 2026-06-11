"""
Verify checkpoint key alignment and frozen-weight integrity across the
Stage 3 pipeline:

  1. Merged checkpoint keys vs model state_dict keys (after detector. strip)
  2. Stage3 trainer checkpoint keys: subset of the full model
  3. After merge-back for eval: frozen keys identical to SAM3 teacher
"""

import argparse
import sys
import torch


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        for key in ("state_dict",):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Cannot extract state_dict")


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def check_merged_vs_model(merged_path, args):
    """Check 1: Do merged checkpoint keys match model state_dict keys?"""
    section("CHECK 1: Merged checkpoint keys vs model state_dict keys")

    sys.path.insert(0, "sam3")
    sys.path.insert(0, ".")
    from sam3.model_builder import build_efficientsam3_image_model

    print(f"  Loading merged checkpoint: {merged_path}")
    merged_sd = _extract_state_dict(_torch_load(merged_path))
    merged_keys = set(merged_sd.keys())

    merged_no_detector = set()
    for k in merged_keys:
        if k.startswith("detector."):
            merged_no_detector.add(k[len("detector."):])
        else:
            merged_no_detector.add(k)

    print(f"  Building model (no checkpoint)...")
    model = build_efficientsam3_image_model(
        bpe_path=args.bpe_path,
        device="cpu",
        eval_mode=True,
        checkpoint_path=None,
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S1",
        text_encoder_context_length=16,
        text_encoder_pos_embed_table_size=16,
        interpolate_pos_embed=False,
    )
    model_keys = set(model.state_dict().keys())

    in_merged_not_model = merged_no_detector - model_keys
    in_model_not_merged = model_keys - merged_no_detector

    print(f"\n  Merged checkpoint keys (after stripping detector.): {len(merged_no_detector)}")
    print(f"  Model state_dict keys: {len(model_keys)}")
    print(f"  In merged but NOT in model: {len(in_merged_not_model)}")
    for k in sorted(in_merged_not_model)[:10]:
        print(f"    {k}")
    print(f"  In model but NOT in merged: {len(in_model_not_merged)}")
    for k in sorted(in_model_not_merged)[:10]:
        print(f"    {k}")

    common = merged_no_detector & model_keys
    shape_mismatch = []
    for k in sorted(common):
        ms = merged_sd.get(f"detector.{k}", merged_sd.get(k))
        mm = model.state_dict()[k]
        if ms.shape != mm.shape:
            shape_mismatch.append((k, ms.shape, mm.shape))

    if shape_mismatch:
        print(f"\n  Shape mismatches: {len(shape_mismatch)}")
        for k, s1, s2 in shape_mismatch[:10]:
            print(f"    {k}: merged={list(s1)} vs model={list(s2)}")
    else:
        print(f"\n  All {len(common)} common keys have matching shapes.")

    ok = len(in_merged_not_model) == 0 and len(shape_mismatch) == 0
    if len(in_model_not_merged) > 0:
        print(f"  NOTE: {len(in_model_not_merged)} model keys not in merged ckpt (e.g. segmentation head, "
              "sam2 convs) -- these are expected when enable_segmentation=False or "
              "components are initialized by model definition.")
    print(f"  {'PASS' if ok else 'ISSUES FOUND'}")
    del model
    return merged_sd


def check_trainer_ckpt_keys(trainer_path, merged_sd):
    """Check 2: Are trainer checkpoint keys a proper subset of merged keys?"""
    section("CHECK 2: Trainer checkpoint keys subset of full model")

    print(f"  Loading trainer checkpoint: {trainer_path}")
    trainer_obj = _torch_load(trainer_path)
    trainer_sd = trainer_obj.get("model", trainer_obj)
    epoch = trainer_obj.get("epoch", "?")
    print(f"  Epoch: {epoch}, Keys: {len(trainer_sd)}")

    merged_no_detector = {}
    for k, v in merged_sd.items():
        nk = k[len("detector."):] if k.startswith("detector.") else k
        merged_no_detector[nk] = v

    not_in_merged = []
    shape_mismatch = []
    for k, v in trainer_sd.items():
        if k not in merged_no_detector:
            not_in_merged.append(k)
        elif v.shape != merged_no_detector[k].shape:
            shape_mismatch.append((k, v.shape, merged_no_detector[k].shape))

    vis_keys = [k for k in trainer_sd if k.startswith("backbone.vision_backbone.trunk.")]
    txt_keys = [k for k in trainer_sd if k.startswith("backbone.language_backbone.")]
    other_keys = [k for k in trainer_sd if k not in set(vis_keys) | set(txt_keys)]

    print(f"\n  Vision encoder keys: {len(vis_keys)}")
    print(f"  Text encoder keys:   {len(txt_keys)}")
    print(f"  Other keys (unexpected): {len(other_keys)}")
    for k in other_keys[:5]:
        print(f"    {k}")

    print(f"  Keys not in merged: {len(not_in_merged)}")
    for k in not_in_merged[:5]:
        print(f"    {k}")
    print(f"  Shape mismatches: {len(shape_mismatch)}")

    ok = len(not_in_merged) == 0 and len(shape_mismatch) == 0 and len(other_keys) == 0
    print(f"  {'PASS' if ok else 'ISSUES FOUND'}")
    return trainer_sd


def check_frozen_identical_to_teacher(merged_path, sam3_path, trainer_path):
    """Check 3: After merge-back, are frozen keys bitwise identical to SAM3?"""
    section("CHECK 3: Frozen keys identical to SAM3 teacher after training+merge")

    print(f"  Loading SAM3 teacher: {sam3_path}")
    teacher_sd = _extract_state_dict(_torch_load(sam3_path))
    teacher_keys = set(teacher_sd.keys())

    print(f"  Loading base merged checkpoint: {merged_path}")
    base_sd = _extract_state_dict(_torch_load(merged_path))

    print(f"  Loading trainer checkpoint: {trainer_path}")
    trainer_obj = _torch_load(trainer_path)
    trainer_sd = trainer_obj.get("model", trainer_obj)

    print(f"  Simulating merge-back (same as merge_stage3_checkpoint_for_eval.py)...")
    eval_sd = dict(base_sd)
    overridden = 0
    for k, v in trainer_sd.items():
        detector_k = f"detector.{k}"
        if detector_k in eval_sd:
            eval_sd[detector_k] = v
            overridden += 1
    print(f"  Overrode {overridden} keys from trainer checkpoint.")

    encoder_prefixes = (
        "detector.backbone.vision_backbone.trunk.",
        "detector.backbone.language_backbone.",
    )

    frozen_in_teacher = {
        k: v for k, v in teacher_sd.items()
        if not any(k.startswith(p) for p in encoder_prefixes)
    }

    print(f"\n  SAM3 teacher total keys: {len(teacher_sd)}")
    print(f"  SAM3 teacher frozen (non-encoder) keys: {len(frozen_in_teacher)}")

    identical = 0
    different = 0
    missing_in_eval = 0
    missing_in_teacher = 0
    diff_keys = []

    for k, t_val in frozen_in_teacher.items():
        if k not in eval_sd:
            missing_in_eval += 1
            continue
        e_val = eval_sd[k]
        if t_val.shape != e_val.shape:
            different += 1
            diff_keys.append((k, "shape", t_val.shape, e_val.shape))
        elif torch.equal(t_val, e_val):
            identical += 1
        else:
            max_diff = (t_val.float() - e_val.float()).abs().max().item()
            different += 1
            diff_keys.append((k, "value", max_diff))

    eval_non_encoder = {
        k: v for k, v in eval_sd.items()
        if not any(k.startswith(p) for p in encoder_prefixes)
    }
    for k in eval_non_encoder:
        if k not in frozen_in_teacher:
            missing_in_teacher += 1

    print(f"\n  Frozen keys identical to SAM3: {identical}")
    print(f"  Frozen keys DIFFERENT from SAM3: {different}")
    print(f"  Frozen keys missing in eval ckpt: {missing_in_eval}")
    print(f"  Eval non-encoder keys missing in SAM3: {missing_in_teacher}")

    if diff_keys:
        print(f"\n  Differences:")
        for item in diff_keys[:20]:
            if item[1] == "shape":
                print(f"    {item[0]}: shape mismatch teacher={list(item[2])} vs eval={list(item[3])}")
            else:
                print(f"    {item[0]}: max_abs_diff={item[2]:.2e}")

    encoder_changed = 0
    encoder_identical = 0
    for prefix in encoder_prefixes:
        teacher_enc = {k: v for k, v in teacher_sd.items() if k.startswith(prefix)}
        for k, t_val in teacher_enc.items():
            if k in eval_sd:
                if torch.equal(t_val, eval_sd[k]):
                    encoder_identical += 1
                else:
                    encoder_changed += 1

    print(f"\n  Encoder keys changed from teacher (expected): {encoder_changed}")
    print(f"  Encoder keys same as teacher (unexpected if trained): {encoder_identical}")

    ok = different == 0 and missing_in_eval == 0
    print(f"\n  {'PASS — all frozen weights are bitwise identical to SAM3' if ok else 'ISSUES FOUND'}")
    return ok


def parse_args():
    parser = argparse.ArgumentParser("Stage 3 Checkpoint Integrity Verification")
    parser.add_argument("--merged-ckpt",
                        default="output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth")
    parser.add_argument("--sam3-ckpt",
                        default="sam3_checkpoints/sam3.pt")
    parser.add_argument("--trainer-ckpt",
                        default="output/stage3/mixed_smoketest_es_tv_m_mc_s1_ctx16/checkpoints/checkpoint.pt")
    parser.add_argument("--bpe-path",
                        default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--skip-model-build", action="store_true",
                        help="Skip building the model (check 1); only compare checkpoints.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_model_build:
        merged_sd = check_merged_vs_model(args.merged_ckpt, args)
    else:
        merged_sd = _extract_state_dict(_torch_load(args.merged_ckpt))

    check_trainer_ckpt_keys(args.trainer_ckpt, merged_sd)
    check_frozen_identical_to_teacher(args.merged_ckpt, args.sam3_ckpt, args.trainer_ckpt)

    section("ALL CHECKS COMPLETE")


if __name__ == "__main__":
    main()
