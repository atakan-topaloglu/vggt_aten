# visualize_attention.py

import torch
import os
import argparse
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import math

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="UserWarning: The given NumPy array is not writable")
warnings.filterwarnings("ignore", "FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated")


def main(args):
    """
    Main function to load images, run VGGT, and save attention maps for all global layers.
    """
    # 1. Setup Device and Directories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Using device: {device} with dtype: {dtype}")

    # Create output directories for attention maps
    global_attn_dir = os.path.join(args.output_dir, "global")
    os.makedirs(global_attn_dir, exist_ok=True)
    print(f"Global attention maps will be saved to: {global_attn_dir}")

    # 2. Load VGGT Model
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.to(device).eval()
    print("Model loaded successfully.")

    # 3. Load and Preprocess Images
    all_image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    if not all_image_paths:
        raise FileNotFoundError(f"No images found in directory: {args.image_dir}")

    if args.num_images is not None:
        if args.num_images < 2:
            raise ValueError("--num_images must be 2 or greater for global attention.")
        all_image_paths = all_image_paths[:args.num_images]

    print(f"Processing {len(all_image_paths)} images: {', '.join([os.path.basename(p) for p in all_image_paths])}")

    # The entire sequence is one "batch" for global attention
    images = load_and_preprocess_images(all_image_paths, mode='pad').to(device)

    # 4. Loop Through All Global Layers and Run Inference
    num_global_layers = 24  # As defined in the VGGT architecture
    print(f"\nStarting iterative visualization for {num_global_layers} global layers...")

    for target_layer in range(num_global_layers):
        print(f"--- Visualizing Global Layer {target_layer:02d} ---")
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=dtype):
                # Pass the current target layer to the model
                _ = model(
                    images,
                    visualize_attn_maps=True,
                    visualize_output_dir=args.output_dir,
                    vis_target_layer=target_layer,
                    vis_source_frame=args.source_frame,
                )

    print("\nVisualization complete for all layers.")
    print(f"Global attention maps saved in: {global_attn_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VGGT to visualize and save attention maps for all global layers.",
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
        default="attention_maps",
        help="Directory where the attention map images will be saved."
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Limit the number of images to process from the directory. Processes all if not set."
    )
    parser.add_argument(
        "--source_frame",
        type=int,
        default=0,
        help="The source frame index for the attention query."
    )

    args = parser.parse_args()
    main(args)