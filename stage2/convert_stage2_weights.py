import torch
import argparse
import os

def convert(args):
    print("="*80)
    print("EfficientSAM3 Final Weight Conversion")
    print("="*80)
    
    # Load all three checkpoints
    print(f"\n[1/5] Loading Stage 1 Vision Student: {args.image_student_checkpoint}")
    image_student_ckpt = torch.load(args.image_student_checkpoint, map_location="cpu", weights_only=False)
    if "model" in image_student_ckpt:
        image_student_state = image_student_ckpt["model"]
    else:
        image_student_state = image_student_ckpt
    print(f"  Loaded {len(image_student_state)} keys")
    
    print(f"\n[2/5] Loading Stage 1 Text Student: {args.text_student_checkpoint}")
    text_student_ckpt = torch.load(args.text_student_checkpoint, map_location="cpu", weights_only=False)
    if "model" in text_student_ckpt:
        text_student_state = text_student_ckpt["model"]
    else:
        text_student_state = text_student_ckpt
    print(f"  Loaded {len(text_student_state)} keys")
    
    print(f"\n[3/5] Loading Stage 2 (Trained Memory + SAM3): {args.stage2_checkpoint}")
    stage2_state = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(stage2_state)} keys")
    
    # Start building the final state dict
    print(f"\n[4/5] Assembling final model...")
    new_state_dict = {}
    
    # 1. Copy trained Hybrid Memory Module from Stage 2
    print("  [a] Copying Hybrid Memory Module from Stage 2...")
    memory_count = 0
    for k, v in stage2_state.items():
        if k.startswith("maskmem_backbone."):
            new_state_dict[k] = v
            memory_count += 1
    print(f"      ✓ Copied {memory_count} memory module keys")
    
    # 2. Copy SAM decoder, FPN neck, and other tracker components from Stage 2
    print("  [b] Copying SAM3 decoder and tracker components from Stage 2...")
    component_prefixes = [
        "sam_mask_decoder.", "sam_prompt_encoder.", "transformer.",
        "mask_downsample.", "maskmem_tpos_enc", "no_mem_embed", 
        "no_mem_pos_enc", "no_obj_ptr", "no_obj_embed_spatial",
        "obj_ptr_proj.", "obj_ptr_tpos_proj.",
        "backbone.vision_backbone.convs.",  # FPN neck
        "backbone.vision_backbone.sam2_convs.",  # SAM2-compatible FPN
        "backbone.vision_backbone.position_encoding.",  # Position encoding for FPN
    ]
    component_count = 0
    for k, v in stage2_state.items():
        if any(k.startswith(prefix) for prefix in component_prefixes):
            new_state_dict[k] = v
            component_count += 1
    print(f"      ✓ Copied {component_count} SAM3 component keys")
    
    # 3. Copy Vision Student Encoder from Stage 1 Image
    # Image student keys are "backbone.model.*" and "head.*"
    # We need to map them to "backbone.vision_backbone.trunk.model.*"
    print("  [c] Copying Vision Student Encoder from Stage 1 Image...")
    vision_count = 0
    for k, v in image_student_state.items():
        if k.startswith("backbone.model."):
            # Map: "backbone.model.X" -> "backbone.vision_backbone.trunk.model.backbone.model.X"
            new_key = k.replace("backbone.model.", "backbone.vision_backbone.trunk.model.backbone.model.")
            new_state_dict[new_key] = v
            vision_count += 1
        elif k.startswith("head."):
            # Map: "head.X" -> "backbone.vision_backbone.trunk.model.head.X"
            new_key = "backbone.vision_backbone.trunk.model." + k
            new_state_dict[new_key] = v
            vision_count += 1
        elif k.startswith("backbone."):
            # Fallback: Map: "backbone.X" -> "backbone.vision_backbone.trunk.model.backbone.X"
            new_key = k.replace("backbone.", "backbone.vision_backbone.trunk.model.backbone.")
            new_state_dict[new_key] = v
            vision_count += 1
        elif k.startswith("model."):
            # Map: "model.X" -> "backbone.vision_backbone.trunk.model.backbone.model.X"
            new_key = "backbone.vision_backbone.trunk.model.backbone." + k
            new_state_dict[new_key] = v
            vision_count += 1
    print(f"      ✓ Copied {vision_count} vision encoder keys")
    
    # 4. Copy Text Student Encoder from Stage 1 Text
    print("  [d] Copying Text Student Encoder from Stage 1 Text...")
    text_count = 0
    for k, v in text_student_state.items():
        # Stage 1 text encoder keys can be:
        # - "encoder.*" (MobileCLIP format)
        # - "projector.*" (output projection)
        # - "backbone.*" 
        # - "model.*"
        # We need to map them to "backbone.language_backbone.*"
        if k.startswith("encoder."):
            # Map: "encoder.X" -> "backbone.language_backbone.encoder.X"
            new_key = "backbone.language_backbone." + k
            new_state_dict[new_key] = v
            text_count += 1
        elif k.startswith("projector."):
            # Map: "projector.X" -> "backbone.language_backbone.projector.X"
            new_key = "backbone.language_backbone." + k
            new_state_dict[new_key] = v
            text_count += 1
        elif k.startswith("backbone."):
            # Map: "backbone.X" -> "backbone.language_backbone.X"
            new_key = k.replace("backbone.", "backbone.language_backbone.")
            new_state_dict[new_key] = v
            text_count += 1
        elif k.startswith("model."):
            # Map: "model.X" -> "backbone.language_backbone.X"
            new_key = k.replace("model.", "backbone.language_backbone.")
            new_state_dict[new_key] = v
            text_count += 1
    print(f"      ✓ Copied {text_count} text encoder keys")
    
    # Verify we have the necessary components
    print(f"\n[5/5] Verification:")
    has_memory = any("maskmem_backbone" in k for k in new_state_dict)
    has_vision = any("backbone.vision_backbone" in k for k in new_state_dict)
    has_text = any("backbone.language_backbone" in k for k in new_state_dict)
    has_decoder = any("sam_mask_decoder" in k for k in new_state_dict)
    has_transformer = any("transformer.encoder" in k for k in new_state_dict)
    
    print(f"  - Hybrid Memory Module: {'✓' if has_memory else '✗'} ({memory_count} keys)")
    print(f"  - Vision Encoder (Student): {'✓' if has_vision else '✗'} ({vision_count} keys)")
    print(f"  - Text Encoder (Student): {'✓' if has_text else '✗'} ({text_count} keys)")
    print(f"  - SAM Mask Decoder: {'✓' if has_decoder else '✗'}")
    print(f"  - Transformer: {'✓' if has_transformer else '✗'}")
    print(f"  - Total keys: {len(new_state_dict)}")
    
    if not (has_memory and has_vision and has_decoder):
        print("\n⚠ WARNING: Missing critical components!")
        return
    else:
        print("\n✓ All critical components present!")
        
    print(f"\nSaving merged model to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(new_state_dict, args.output_path)
    print(f"✓ Saved successfully!")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge Stage 1 student encoders with Stage 2 trained memory into final EfficientSAM3 model"
    )
    parser.add_argument("--image_student_checkpoint", type=str, required=True,
                        help="Path to Stage 1 trained vision encoder (e.g., output/stage1/es_ev_s/ckpt_epoch_0.pth)")
    parser.add_argument("--text_student_checkpoint", type=str, required=True,
                        help="Path to Stage 1 trained text encoder (e.g., output/stage1_text/mobileclip_s/ckpt_epoch_0.pth)")
    parser.add_argument("--stage2_checkpoint", type=str, required=True,
                        help="Path to Stage 2 trained memory module (e.g., output/stage2/epoch_2.pth)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save final merged model (e.g., output/efficient_sam3_final.pt)")
    args = parser.parse_args()
    convert(args)
