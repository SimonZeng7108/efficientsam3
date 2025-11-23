#!/usr/bin/env python3
"""
Interactive Instance Segmentation using EfficientSAM3

Segment Anything Model 3 (SAM 3) with EfficientViT backbone predicts instance masks 
that indicate the desired object given geometric prompts (SAM 1 task).

The `SAM3Image` and `Sam3Processor` classes provide an easy interface to prompt the model. 
The user first sets an image using the `Sam3Processor.set_image` method, which computes 
the necessary image embeddings. Then, prompts can be provided via the `predict` method 
to efficiently predict masks from those prompts. The model can take as input both point 
and box prompts, as well as masks from the previous iteration of prediction.

This script follows the SAM 2 API for interactive image segmentation using EfficientSAM3.
"""

import os
import sys
# Add the parent directory to sys.path to allow importing sam3
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sam3

# Set environment variables
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

# Select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# Setup device-specific optimizations
autocast_context = None
if device.type == "cuda":
    # use bfloat16 for the entire script
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    autocast_context.__enter__()
    # turn on tfloat32 for Ampere GPUs
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

# Set random seed for reproducibility
np.random.seed(3)


def show_mask(mask, ax, random_color=False, borders=True):
    """Display a mask on the given axes."""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """Display points on the given axes."""
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    """Display a bounding box on the given axes."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                                facecolor=(0, 0, 0, 0), lw=2))


def show_masks(image, masks, scores, point_coords=None, box_coords=None, 
               input_labels=None, borders=True, save_path=None):
    """Display masks with optional points and boxes."""
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, "../assets")

    parser = argparse.ArgumentParser(description='EfficientSAM3 Interactive Instance Segmentation')
    parser.add_argument('--checkpoint', type=str, 
                       default='output/efficient_sam3_efficientvit_b0.pt', #
                       help='Path to EfficientSAM3 checkpoint')
    parser.add_argument('--image', type=str, 
                       default=os.path.join(assets_dir, 'images/truck.jpg'),
                       help='Path to input image')
    parser.add_argument('--bpe_path', type=str,
                       default=os.path.join(assets_dir, 'bpe_simple_vocab_16e6.txt.gz'),
                       help='Path to BPE tokenizer vocabulary')
    parser.add_argument('--load_from_hf', action='store_true',
                       help='Load checkpoint from HuggingFace')
    parser.add_argument('--demo', type=str, default='single_point',
                       choices=['single_point', 'multiple_points', 'box', 
                               'combined', 'batched_boxes', 'batched_images'],
                       help='Which demo to run')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display)')
    parser.add_argument('--backbone_type', type=str, default='efficientvit', help='Backbone type')
    parser.add_argument('--model_name', type=str, default='b0', help='Model name')
    args = parser.parse_args()

    if args.headless:
        import matplotlib.pyplot as plt
        plt.show = lambda: None
        print("Running in headless mode - display disabled")

    # Load the image
    print(f"Loading image from: {args.image}")
    image = Image.open(args.image)
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.title("Input Image")
    # plt.show()

    # Load the EfficientSAM3 model
    print("Loading EfficientSAM3 model...")
    from sam3.model_builder import build_efficientsam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe_path = args.bpe_path
    model = build_efficientsam3_image_model(
        bpe_path=bpe_path,
        enable_inst_interactivity=True,
        checkpoint_path=args.checkpoint,
        load_from_HF=args.load_from_hf,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
    )
    print("Model loaded successfully!")

    # Process the image to produce an image embedding
    print("Processing image...")
    processor = Sam3Processor(model)
    inference_state = processor.set_image(image)
    print("Image processed!")

    # Run different demos based on argument
    if args.demo == 'single_point':
        # Single point prompt
        print("\n=== Single Point Prompt Demo ===")
        input_point = np.array([[520, 375]])
        input_label = np.array([1])
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_points(input_point, input_label, plt.gca())
        plt.axis('on')
        plt.title("Input Point")
        plt.show()

        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        
        print(f"Masks shape: {masks.shape}")
        show_masks(image, masks, scores, point_coords=input_point, 
                  input_labels=input_label, borders=True)

    elif args.demo == 'multiple_points':
        # Multiple points prompt
        print("\n=== Multiple Points Prompt Demo ===")
        input_point = np.array([[500, 375], [1125, 625]])
        input_label = np.array([1, 1])
        
        # Use previous best mask as input
        masks_prev, scores_prev, logits_prev = model.predict_inst(
            inference_state,
            point_coords=np.array([[520, 375]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        mask_input = logits_prev[np.argmax(scores_prev), :, :]
        
        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        
        print(f"Masks shape: {masks.shape}")
        show_masks(image, masks, scores, point_coords=input_point, 
                  input_labels=input_label)

    elif args.demo == 'box':
        # Box prompt
        print("\n=== Box Prompt Demo ===")
        input_box = np.array([425, 600, 700, 875])
        
        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        
        show_masks(image, masks, scores, box_coords=input_box)

    elif args.demo == 'combined':
        # Combined points and box
        print("\n=== Combined Points and Box Demo ===")
        input_box = np.array([425, 600, 700, 875])
        input_point = np.array([[575, 750]])
        input_label = np.array([0])
        
        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )
        
        show_masks(image, masks, scores, box_coords=input_box, 
                  point_coords=input_point, input_labels=input_label)

    elif args.demo == 'batched_boxes':
        # Batched box prompts
        print("\n=== Batched Box Prompts Demo ===")
        input_boxes = np.array([
            [75, 275, 1725, 850],
            [425, 600, 700, 875],
            [1375, 550, 1650, 800],
            [1240, 675, 1400, 750],
        ])
        
        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        print(f"Masks shape: {masks.shape}")
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.squeeze(0), plt.gca(), random_color=True)
        for box in input_boxes:
            show_box(box, plt.gca())
        plt.axis('off')
        plt.title("Batched Box Predictions")
        plt.show()

    elif args.demo == 'batched_images':
        # Batched images with boxes
        print("\n=== Batched Images Demo ===")
        image1 = image
        image1_boxes = np.array([
            [75, 275, 1725, 850],
            [425, 600, 700, 875],
            [1375, 550, 1650, 800],
            [1240, 675, 1400, 750],
        ])
        
        image2_path = f"{sam3_root}/assets/images/groceries.jpg"
        if os.path.exists(image2_path):
            image2 = Image.open(image2_path)
            image2_boxes = np.array([
                [450, 170, 520, 350],
                [350, 190, 450, 350],
                [500, 170, 580, 350],
                [580, 170, 640, 350],
            ])
            
            img_batch = [image1, image2]
            boxes_batch = [image1_boxes, image2_boxes]
            
            inference_state = processor.set_image_batch(img_batch)
            
            masks_batch, scores_batch, _ = model.predict_inst_batch(
                inference_state,
                None,
                None,
                box_batch=boxes_batch,
                multimask_output=False
            )
            
            for img, boxes, masks in zip(img_batch, boxes_batch, masks_batch):
                plt.figure(figsize=(10, 10))
                plt.imshow(img)
                for mask in masks:
                    show_mask(mask.squeeze(0), plt.gca(), random_color=True)
                for box in boxes:
                    show_box(box, plt.gca())
                plt.axis('off')
                plt.show()
        else:
            print(f"Warning: {image2_path} not found. Skipping batched images demo.")

    print("\nDemo completed!")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up autocast context if it was created
        if autocast_context is not None:
            autocast_context.__exit__(None, None, None)

