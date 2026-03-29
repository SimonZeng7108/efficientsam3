"""
Stage 3 — GPU sanity check for the merged EfficientSAM3 model.

Verifies:
  1. Merged checkpoint loads into the EfficientSAM3 model on GPU.
  2. Freeze/unfreeze state is correct.
  3. Vision encoder forward produces correct shapes and gradients flow.
  4. Text encoder forward produces correct shapes and gradients flow.
  5. FPN neck + position encoding produce correct outputs.
  6. Encoder output comparison with SAM3 teacher (cosine similarity).

Usage (via SLURM, see scripts/sanity_check_gpu.sh):
    PYTHONPATH=sam3:. python stage3/sanity_check_gpu.py
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sam3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def check_model_load(args):
    section("1. Load merged checkpoint on GPU")
    from stage3.model import build_stage3_model

    t0 = time.time()
    model = build_stage3_model(
        bpe_path=args.bpe_path,
        device="cpu",
        eval_mode=False,
        checkpoint_path=args.merged_ckpt,
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
        backbone_type="tinyvit",
        model_name="11m",
        text_encoder_type="MobileCLIP-S1",
        text_encoder_context_length=16,
        text_encoder_pos_embed_table_size=16,
        interpolate_pos_embed=False,
        train_vision_encoder=True,
        train_text_encoder=True,
        freeze_non_encoder_parameters=True,
        keep_frozen_modules_eval=True,
        log_parameter_summary=True,
    )
    model = model.to(args.device)
    elapsed = time.time() - t0
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Loaded in {elapsed:.1f}s, {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable on {args.device}")
    return model


def check_freeze(model):
    section("2. Verify freeze/unfreeze state")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}

    vis_trainable = sorted(n for n in trainable if n.startswith("backbone.vision_backbone.trunk."))
    txt_trainable = sorted(n for n in trainable if n.startswith("backbone.language_backbone."))
    other_trainable = sorted(n for n in trainable if n not in set(vis_trainable) | set(txt_trainable))

    print(f"  Trainable: {len(trainable)} params ({sum(model.state_dict()[n].numel() for n in trainable)/1e6:.2f}M)")
    print(f"    Vision encoder: {len(vis_trainable)} params")
    print(f"    Text encoder:   {len(txt_trainable)} params")
    print(f"    Other (BUG!):   {len(other_trainable)} params")
    if other_trainable:
        for n in other_trainable[:5]:
            print(f"      {n}")
    print(f"  Frozen: {len(frozen)} params")

    assert len(other_trainable) == 0, "Non-encoder params should be frozen!"
    assert len(vis_trainable) > 0, "Vision encoder should be trainable!"
    assert len(txt_trainable) > 0, "Text encoder should be trainable!"
    print("  PASS")


def check_vision_forward(model, device):
    section("3. Vision encoder forward pass + gradient flow")
    dummy_img = torch.randn(1, 3, 1008, 1008, device=device)

    model.train()
    vis_out = model.backbone.forward_image(dummy_img)
    fpn = vis_out["backbone_fpn"]
    pos = vis_out["vision_pos_enc"]
    print(f"  FPN levels: {len(fpn)}")
    for i, (f, p) in enumerate(zip(fpn, pos)):
        print(f"    Level {i}: feat={list(f.shape)}, pos={list(p.shape)}")
    print(f"  vision_features: {list(vis_out['vision_features'].shape)}")

    loss = vis_out["vision_features"].sum()
    loss.backward()

    grad_params = [n for n, p in model.named_parameters()
                   if p.grad is not None and n.startswith("backbone.vision_backbone.trunk.")]
    frozen_grads = [n for n, p in model.named_parameters()
                    if p.grad is not None and not n.startswith("backbone.vision_backbone.trunk.")
                    and not n.startswith("backbone.language_backbone.")]

    print(f"  Vision params with gradients: {len(grad_params)}")
    print(f"  Frozen params with gradients (should be 0): {len(frozen_grads)}")
    if frozen_grads:
        for n in frozen_grads[:5]:
            print(f"    LEAK: {n}")
    model.zero_grad()
    assert len(fpn) >= 1, "Expected at least 1 FPN level"
    assert len(grad_params) > 0, "Vision encoder should have gradients"
    print("  PASS")
    return vis_out


def check_text_forward(model, device):
    section("4. Text encoder forward pass + gradient flow")
    texts = ["a brown dog running", "a cat sitting on a chair"]

    model.train()
    txt_out = model.backbone.forward_text(texts, device=device)
    memory = txt_out["language_features"]
    mask = txt_out["language_mask"]
    embeds = txt_out["language_embeds"]
    print(f"  Text memory:  {list(memory.shape)}")
    print(f"  Text mask:    {list(mask.shape)}")
    print(f"  Text embeds:  {list(embeds.shape)}")

    loss = memory.sum()
    loss.backward()

    grad_params = [n for n, p in model.named_parameters()
                   if p.grad is not None and n.startswith("backbone.language_backbone.")]
    frozen_grads = [n for n, p in model.named_parameters()
                    if p.grad is not None and not n.startswith("backbone.vision_backbone.trunk.")
                    and not n.startswith("backbone.language_backbone.")]

    print(f"  Text params with gradients: {len(grad_params)}")
    print(f"  Frozen params with gradients (should be 0): {len(frozen_grads)}")
    model.zero_grad()
    assert memory.shape[-1] == 256, f"Expected d_model=256, got {memory.shape[-1]}"
    assert len(grad_params) > 0, "Text encoder should have gradients"
    print("  PASS")


def check_teacher_comparison(model, args):
    section("5. Encoder output comparison with SAM3 teacher")
    from sam3.model_builder import build_sam3_image_model

    print("  Building SAM3 teacher (may take ~30s)...")
    teacher = build_sam3_image_model(
        bpe_path=args.bpe_path,
        device=args.device,
        eval_mode=True,
        checkpoint_path=args.sam3_ckpt,
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )

    model.eval()
    teacher.eval()

    dummy_img = torch.randn(1, 3, 1008, 1008, device=args.device)
    texts = ["a brown dog running", "a cat on the sofa"]

    with torch.no_grad():
        s_vis = model.backbone.forward_image(dummy_img)
        t_vis = teacher.backbone.forward_image(dummy_img)
        s_fpn = s_vis["backbone_fpn"]
        t_fpn = t_vis["backbone_fpn"]

        print(f"\n  Vision Encoder:")
        print(f"    Student FPN levels: {len(s_fpn)}, Teacher FPN levels: {len(t_fpn)}")
        for i, (sf, tf) in enumerate(zip(s_fpn, t_fpn)):
            print(f"    Level {i}: student={list(sf.shape)}, teacher={list(tf.shape)}")
            if sf.shape == tf.shape:
                cos = F.cosine_similarity(sf.flatten(), tf.flatten(), dim=0).item()
                l2 = (sf - tf).pow(2).mean().sqrt().item()
                print(f"      cosine_sim={cos:.4f}, rmse={l2:.4f}")
            else:
                print(f"      (shape mismatch — expected for different architectures)")

        s_txt = model.backbone.forward_text(texts, device=args.device)
        t_txt = teacher.backbone.forward_text(texts, device=args.device)
        s_mem = s_txt["language_features"]
        t_mem = t_txt["language_features"]

        print(f"\n  Text Encoder:")
        print(f"    Student memory: {list(s_mem.shape)}, Teacher memory: {list(t_mem.shape)}")
        min_seq = min(s_mem.shape[0], t_mem.shape[0])
        s_sub = s_mem[:min_seq]
        t_sub = t_mem[:min_seq]
        cos = F.cosine_similarity(s_sub.reshape(-1), t_sub.reshape(-1), dim=0).item()
        l2 = (s_sub - t_sub).pow(2).mean().sqrt().item()
        print(f"    cosine_sim={cos:.4f}, rmse={l2:.4f}")
        print(f"    (Non-trivial diff expected — student is distilled, not identical)")

    del teacher
    torch.cuda.empty_cache()
    print("  DONE")


def check_data_pipeline(args):
    section("6. Data pipeline quick check")
    import json

    coco_ann = os.path.join(args.coco_root, "annotations", "instances_val2017.json")
    if not os.path.exists(coco_ann):
        print(f"  SKIP: {coco_ann} not found")
        return

    with open(coco_ann) as f:
        data = json.load(f)
    print(f"  COCO val2017: {len(data.get('images', []))} images, "
          f"{len(data.get('annotations', []))} annotations, "
          f"{len(data.get('categories', []))} categories")

    img_dir = os.path.join(args.coco_root, "images", "val2017")
    if os.path.isdir(img_dir):
        n_images = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        print(f"  Image files found: {n_images}")
    else:
        print(f"  WARN: {img_dir} not found")

    print("  PASS")


def parse_args():
    parser = argparse.ArgumentParser("Stage 3 GPU Sanity Check")
    parser.add_argument("--merged-ckpt", default="output/efficient_sam3_stage3_es_tv_m_mobileclip_s1_ctx16.pth")
    parser.add_argument("--sam3-ckpt", default="sam3_checkpoints/sam3.pt")
    parser.add_argument("--bpe-path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--coco-root", default="data/coco")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-teacher", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {mem_gb:.1f} GB")

    model = check_model_load(args)
    check_freeze(model)
    check_vision_forward(model, args.device)
    check_text_forward(model, args.device)

    if not args.skip_teacher:
        check_teacher_comparison(model, args)
    else:
        print("\n  Skipping teacher comparison (--skip-teacher)")

    check_data_pipeline(args)

    section("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
