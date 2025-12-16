"""
Debug script to compare teacher and student text encoder outputs.

This script verifies:
1. Tokenization is the same for both teacher and student
2. Teacher embeddings (saved during save_text_embeddings.sh) match live teacher output
3. Student output after training matches teacher output (distillation target)
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.text_encoder_student import TextStudentEncoderWithTeacherEmbed
from sam3.model.tokenizer_ve import SimpleTokenizer


def load_teacher_encoder(checkpoint_path: str, bpe_path: str, device: str = "cpu"):
    """Load the SAM3 teacher text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    encoder = VETextEncoder(
        d_model=256,  # Output dimension
        tokenizer=tokenizer,
        width=1024,
        heads=16,
        layers=24,
        context_length=32,
        vocab_size=49408,
    )
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    
    # Extract language_backbone weights
    prefix = "detector.backbone.language_backbone."
    encoder_sd = {}
    for k, v in ckpt.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            encoder_sd[new_key] = v
    
    encoder.load_state_dict(encoder_sd, strict=False)
    encoder = encoder.to(device).eval()
    return encoder


def load_student_encoder(checkpoint_path: str, bpe_path: str, sam3_checkpoint: str, device: str = "cpu"):
    """Load the trained student text encoder with teacher embeddings."""
    cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "dim": 512,
        "ffn_multiplier_per_layer": 4.0,
        "n_heads_per_layer": 8,
        "n_transformer_layers": 4,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "model_name": "mct",
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
    }
    
    encoder = TextStudentEncoderWithTeacherEmbed(
        cfg=cfg,
        context_length=32,
        output_dim=256,
        teacher_embed_dim=1024,
        bpe_path=bpe_path,
    )
    
    # Load teacher embeddings first
    encoder.load_teacher_embeddings(sam3_checkpoint)
    
    # Load student checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    
    # Load student-specific weights (embed_proj, encoder, projector)
    missing, unexpected = encoder.load_state_dict(ckpt, strict=False)
    print(f"Student loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    
    encoder = encoder.to(device).eval()
    return encoder


def compare_tokenization(teacher: VETextEncoder, student: TextStudentEncoderWithTeacherEmbed, text: str):
    """Compare tokenization between teacher and student."""
    print(f"\n{'='*60}")
    print(f"Comparing tokenization for: '{text}'")
    print(f"{'='*60}")
    
    # Teacher tokenization
    teacher_tokens = teacher.tokenizer.encode(text)
    print(f"Teacher tokens: {teacher_tokens}")
    print(f"Teacher token count: {len(teacher_tokens)}")
    
    # Student tokenization
    student_tokens = student.tokenizer.encode(text)
    print(f"Student tokens: {student_tokens}")
    print(f"Student token count: {len(student_tokens)}")
    
    if teacher_tokens == student_tokens:
        print("✓ Tokenization matches!")
    else:
        print("✗ Tokenization MISMATCH!")
    
    return teacher_tokens, student_tokens


def compare_embeddings(teacher: VETextEncoder, student: TextStudentEncoderWithTeacherEmbed, text: str, device: str = "cpu"):
    """Compare raw token embeddings (before transformer)."""
    print(f"\n{'='*60}")
    print(f"Comparing embeddings for: '{text}'")
    print(f"{'='*60}")
    
    with torch.no_grad():
        # Get teacher embedding
        teacher_tokens = teacher.tokenizer.encode(text)
        teacher_token_tensor = torch.tensor([teacher_tokens], device=device)
        
        # Teacher uses context_length=32 for truncation/padding
        if teacher_token_tensor.shape[1] > 32:
            teacher_token_tensor = teacher_token_tensor[:, :32]
        elif teacher_token_tensor.shape[1] < 32:
            padding = torch.zeros(1, 32 - teacher_token_tensor.shape[1], dtype=torch.long, device=device)
            teacher_token_tensor = torch.cat([teacher_token_tensor, padding], dim=1)
        
        # Get raw embeddings from teacher
        teacher_token_emb = teacher.encoder.token_embedding(teacher_token_tensor)
        teacher_pos_emb = teacher.encoder.positional_embedding[:32]
        teacher_combined = teacher_token_emb + teacher_pos_emb
        
        print(f"Teacher token embedding shape: {teacher_token_emb.shape}")
        print(f"Teacher positional embedding shape: {teacher_pos_emb.shape}")
        print(f"Teacher combined embedding shape: {teacher_combined.shape}")
        
        # Get raw embeddings from student (should be same as teacher since we copy them)
        student_tokens = student.tokenizer.encode(text)
        student_token_tensor = torch.tensor([student_tokens], device=device)
        
        if student_token_tensor.shape[1] > 32:
            student_token_tensor = student_token_tensor[:, :32]
        elif student_token_tensor.shape[1] < 32:
            padding = torch.zeros(1, 32 - student_token_tensor.shape[1], dtype=torch.long, device=device)
            student_token_tensor = torch.cat([student_token_tensor, padding], dim=1)
        
        student_token_emb = student.token_embedding(student_token_tensor)
        student_pos_emb = student.positional_embedding
        student_combined = student_token_emb + student_pos_emb
        
        print(f"\nStudent token embedding shape: {student_token_emb.shape}")
        print(f"Student positional embedding shape: {student_pos_emb.shape}")
        print(f"Student combined embedding shape: {student_combined.shape}")
        
        # Compare
        token_emb_diff = (teacher_token_emb - student_token_emb).abs().mean().item()
        pos_emb_diff = (teacher_pos_emb - student_pos_emb).abs().mean().item()
        combined_diff = (teacher_combined - student_combined).abs().mean().item()
        
        print(f"\nToken embedding MAE: {token_emb_diff:.6f}")
        print(f"Positional embedding MAE: {pos_emb_diff:.6f}")
        print(f"Combined embedding MAE: {combined_diff:.6f}")
        
        if token_emb_diff < 1e-5:
            print("✓ Token embeddings match!")
        else:
            print("✗ Token embeddings MISMATCH!")
            
        if pos_emb_diff < 1e-5:
            print("✓ Positional embeddings match!")
        else:
            print("✗ Positional embeddings MISMATCH!")
        
        return teacher_combined, student_combined


def compare_saved_embeddings(saved_path: str, teacher: VETextEncoder, text: str, device: str = "cpu"):
    """Compare saved teacher embeddings with live teacher output."""
    print(f"\n{'='*60}")
    print(f"Comparing saved embeddings for: '{text}'")
    print(f"{'='*60}")
    
    # Load saved embedding using the binary format
    keys_file = os.path.join(saved_path, "rank0-keys.txt")
    values_file = os.path.join(saved_path, "rank0-values.bin")
    
    if not os.path.exists(keys_file) or not os.path.exists(values_file):
        print(f"Saved embedding files not found!")
        return None, None
    
    # Read keys
    with open(keys_file, 'r') as f:
        keys = [line.strip() for line in f.readlines()]
    print(f"Saved keys: {keys}")
    
    # Read values  
    # Format: 4 bytes seed + (embed_dim * num_tokens * 2 bytes float16)
    # For text embeddings: shape [32, 256] = 32 * 256 = 8192 float16 values
    embed_dim = 256
    num_tokens = 32
    num_embeddings = num_tokens  # seq length
    item_size = 4 + embed_dim * 2 * num_embeddings  # 4 bytes seed + embeddings
    
    with open(values_file, 'rb') as f:
        bstr = f.read()
    
    print(f"Binary file size: {len(bstr)} bytes")
    print(f"Expected item size: {item_size} bytes")
    
    if len(bstr) >= item_size:
        # Parse seed
        seed = int(np.frombuffer(bstr[:4], dtype=np.int32))
        print(f"Saved seed: {seed}")
        
        # Parse embeddings
        embeddings_bytes = bstr[4:4 + embed_dim * 2 * num_embeddings]
        saved_embedding = np.frombuffer(embeddings_bytes, dtype=np.float16).copy()
        saved_embedding = saved_embedding.reshape(num_embeddings, embed_dim)
        saved_embedding = torch.from_numpy(saved_embedding.astype(np.float32)).unsqueeze(0)
        
        print(f"Saved embedding shape: {saved_embedding.shape}")
        print(f"Saved embedding stats: min={saved_embedding.min():.4f}, max={saved_embedding.max():.4f}, mean={saved_embedding.mean():.4f}")
    else:
        print(f"Binary file too small!")
        return None, None
    
    # Get live teacher output
    # Teacher returns (attention_mask, text_memory_resized, inputs_embeds)
    with torch.no_grad():
        _, live_embedding, _ = teacher([text])
        live_embedding = live_embedding.transpose(0, 1)  # [seq, batch, dim] -> [batch, seq, dim]
    
    print(f"Live teacher output shape: {live_embedding.shape}")
    print(f"Live teacher stats: min={live_embedding.min():.4f}, max={live_embedding.max():.4f}, mean={live_embedding.mean():.4f}")
    
    # Compare
    mae = (saved_embedding - live_embedding).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        saved_embedding.flatten().unsqueeze(0),
        live_embedding.flatten().unsqueeze(0)
    ).item()
    
    print(f"\nMAE between saved and live: {mae:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    
    if mae < 1e-3:
        print("✓ Saved embeddings match live teacher!")
    else:
        print("✗ Saved embeddings MISMATCH with live teacher!")
    
    return saved_embedding, live_embedding


def compare_final_outputs(teacher: VETextEncoder, student: TextStudentEncoderWithTeacherEmbed, text: str, device: str = "cpu"):
    """Compare final outputs (distillation target vs student output)."""
    print(f"\n{'='*60}")
    print(f"Comparing final outputs for: '{text}'")
    print(f"{'='*60}")
    
    with torch.no_grad():
        # Teacher output - returns (attention_mask, text_memory_resized, inputs_embeds)
        _, teacher_out, _ = teacher([text])
        teacher_out = teacher_out.transpose(0, 1)  # [seq, batch, dim] -> [batch, seq, dim]
        print(f"Teacher output shape: {teacher_out.shape}")
        print(f"Teacher output stats: min={teacher_out.min():.4f}, max={teacher_out.max():.4f}, mean={teacher_out.mean():.4f}")
        
        # Student output - also returns (attention_mask, text_memory, inputs_embeds)
        _, student_out, _ = student([text])
        student_out = student_out.transpose(0, 1)  # [seq, batch, dim] -> [batch, seq, dim]
        print(f"Student output shape: {student_out.shape}")
        print(f"Student output stats: min={student_out.min():.4f}, max={student_out.max():.4f}, mean={student_out.mean():.4f}")
        
        # Compare
        mae = (teacher_out - student_out).abs().mean().item()
        mse = ((teacher_out - student_out) ** 2).mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            teacher_out.reshape(1, -1),
            student_out.reshape(1, -1)
        ).item()
        
        print(f"\nMAE: {mae:.6f}")
        print(f"MSE: {mse:.6f}")
        print(f"Cosine similarity: {cos_sim:.6f}")
        
        # Per-token cosine similarity
        B, S, D = teacher_out.shape
        per_token_cos = []
        for i in range(S):
            cos = torch.nn.functional.cosine_similarity(
                teacher_out[0, i:i+1],
                student_out[0, i:i+1]
            ).item()
            per_token_cos.append(cos)
        
        print(f"\nPer-token cosine similarities (first 10):")
        for i, cos in enumerate(per_token_cos[:10]):
            print(f"  Token {i}: {cos:.4f}")
        
        return teacher_out, student_out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Debug text encoder outputs")
    parser.add_argument("--sam3-ckpt", type=str, default="sam3_checkpoints/sam3.pt",
                       help="Path to SAM3 teacher checkpoint")
    parser.add_argument("--student-ckpt", type=str, 
                       default="output/stage1_text_student_shoe_word_only/ckpt_epoch_499.pth",
                       help="Path to trained student checkpoint")
    parser.add_argument("--saved-embeddings", type=str,
                       default="output/stage1_text_teacher_shoe_word_only/embeddings",
                       help="Path to saved teacher embeddings")
    parser.add_argument("--bpe-path", type=str, 
                       default="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                       help="Path to BPE vocab")
    parser.add_argument("--text", type=str, default="shoe",
                       help="Text to test")
    parser.add_argument("--device", type=str, default="cpu")  # Use CPU to avoid device issues
    args = parser.parse_args()
    
    print("="*60)
    print("TEXT ENCODER DEBUG SCRIPT")
    print("="*60)
    print(f"SAM3 checkpoint: {args.sam3_ckpt}")
    print(f"Student checkpoint: {args.student_ckpt}")
    print(f"Saved embeddings: {args.saved_embeddings}")
    print(f"Test text: '{args.text}'")
    print(f"Device: {args.device}")
    
    # Load teacher
    print("\n[1/5] Loading teacher encoder...")
    teacher = load_teacher_encoder(args.sam3_ckpt, args.bpe_path, args.device)
    print(f"Teacher loaded successfully")
    
    # Load student
    print("\n[2/5] Loading student encoder...")
    student = load_student_encoder(args.student_ckpt, args.bpe_path, args.sam3_ckpt, args.device)
    print(f"Student loaded successfully")
    
    # Compare tokenization
    print("\n[3/5] Comparing tokenization...")
    compare_tokenization(teacher, student, args.text)
    
    # Compare embeddings
    print("\n[4/5] Comparing embeddings...")
    compare_embeddings(teacher, student, args.text, args.device)
    
    # Compare saved embeddings with live
    print("\n[5/5] Comparing saved embeddings...")
    if os.path.exists(args.saved_embeddings):
        compare_saved_embeddings(args.saved_embeddings, teacher, args.text, args.device)
    else:
        print(f"Saved embeddings not found at {args.saved_embeddings}")
    
    # Compare final outputs
    print("\n[6/6] Comparing final outputs (distillation target)...")
    compare_final_outputs(teacher, student, args.text, args.device)
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
