# visualize_attention.py

import torch
import os
import argparse
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="UserWarning: The given NumPy array is not writable")

def main(args):
    """
    Main function to load images, run VGGT, and save attention maps.
    """
    # 1. Setup Device and Directories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device} with dtype: {dtype}")

    # Create output directories for attention maps
    frame_attn_dir = os.path.join(args.output_dir, "frame")
    global_attn_dir = os.path.join(args.output_dir, "global")
    os.makedirs(frame_attn_dir, exist_ok=True)
    os.makedirs(global_attn_dir, exist_ok=True)
    print(f"Attention maps will be saved to: {args.output_dir}")

    # 2. Load VGGT Model
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.to(device).eval()
    print("Model loaded successfully.")

    # 3. Load and Preprocess Images
    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in directory: {args.image_dir}")

    # Limit the number of images if specified
    if args.num_images is not None:
        if args.num_images < 2:
            raise ValueError("--num_images must be 2 or greater for global attention.")
        image_paths = image_paths[:args.num_images]

    print(f"Processing {len(image_paths)} images: {', '.join([os.path.basename(p) for p in image_paths])}")
    
    # Using the 'pad' mode to keep aspect ratio and pad to a square
    # This is often better for preserving geometric details.
    images = load_and_preprocess_images(image_paths, mode='pad').to(device)

    # 4. Run Inference with Visualization Enabled
    print("\nRunning inference to generate attention maps...")
    print("This will be slow and may consume significant disk space.")
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # The `visualize_attn_maps` flag triggers the saving logic
            # inside the modified Aggregator module.
            # The model's output dictionary is not needed for this script's purpose.
            _ = model(images, visualize_attn_maps=True, visualize_output_dir=args.output_dir)

    print("\nVisualization complete.")
    print(f"Frame-wise attention maps saved in: {frame_attn_dir}")
    print(f"Global attention maps saved in: {global_attn_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VGGT to visualize and save attention maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True, 
        help="Directory containing the input images."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="attention_maps/attn_maps", 
        help="Directory where the attention map images will be saved."
    )
    parser.add_argument(
        "--num_images", 
        type=int, 
        default=None, 
        help="Limit the number of images to process from the directory. Processes all if not set."
    )

    args = parser.parse_args()
    main(args)