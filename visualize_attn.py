# visualize_attention.py

import torch
import os
import argparse
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="UserWarning: The given NumPy array is not writable")
warnings.filterwarnings("ignore", "FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated")

def normalize_attention_map(raw_attn_map: torch.Tensor, P: int, patch_size: int, patch_start_idx: int, H: int, W: int) -> np.ndarray:
    """
    Normalizes a raw attention map to the [0, 1] range based on its patch token scores.
    Returns ONLY the normalized patch token scores.
    """
    # Squeeze out head and query dimensions, convert to numpy
    attn_map_avg = raw_attn_map.mean(dim=1).squeeze(1).detach().cpu().numpy() # Shape: (B, S*P)
    B, S_times_P = attn_map_avg.shape
    S = S_times_P // P

    patch_h = H // patch_size
    patch_w = W // patch_size
    num_patch_tokens = patch_h * patch_w

    # Isolate the attention scores corresponding ONLY to patch tokens.
    patch_attn_scores = []
    for s_idx in range(S):
        start_idx = s_idx * P + patch_start_idx
        end_idx = start_idx + num_patch_tokens
        patch_attn_scores.append(attn_map_avg[:, start_idx:end_idx])
    patch_only_attn_map = np.concatenate(patch_attn_scores, axis=1)

    # Calculate the global min and max on the relevant patch token scores only.
    global_min = patch_only_attn_map.min()
    global_max = patch_only_attn_map.max()
    global_range = global_max - global_min + 1e-8

    # Normalize and return ONLY the patch token scores.
    return (patch_only_attn_map - global_min) / global_range

def save_attention_maps(normalized_avg_patch_map: np.ndarray, B: int, S: int, patch_size: int, output_dir: str, images: torch.Tensor, source_frame_idx: int, title_prefix: str):
    """
    Saves a pre-normalized map of patch attentions as grayscale images and a colorized grid.
    """
    patch_h = images.shape[-2] // patch_size
    patch_w = images.shape[-1] // patch_size
    num_patch_tokens = patch_h * patch_w

    for b in range(B):
        num_cols = int(math.ceil(math.sqrt(S)))
        num_rows = int(math.ceil(S / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4), squeeze=False)
        fig.suptitle(f'{title_prefix} from Frame {source_frame_idx}', fontsize=16)

        for i in range(num_rows * num_cols):
            row, col = i // num_cols, i % num_cols
            ax = axes[row, col]

            if i < S:
                tgt_frame_idx = i
                # Slice the pre-normalized, patch-only map
                start = tgt_frame_idx * num_patch_tokens
                end = start + num_patch_tokens
                attn_slice_norm = normalized_avg_patch_map[b, start:end].reshape(patch_h, patch_w)

                original_img_tensor = images[b, tgt_frame_idx]
                original_img_np = (original_img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

                # Save grayscale map (already normalized, just scale to 255)
                grayscale_map = (attn_slice_norm * 255).astype(np.uint8)
                gray_fpath = os.path.join(output_dir, "global", f"{title_prefix}_from_frame_{source_frame_idx}_to_frame_{tgt_frame_idx}_gray.png")
                cv2.imwrite(gray_fpath, grayscale_map)

                # Create color overlay for grid
                attn_heatmap_resized = cv2.resize(attn_slice_norm, (original_img_np.shape[1], original_img_np.shape[0]))
                attn_heatmap_colored = (plt.cm.plasma(attn_heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
                attn_heatmap_bgr = cv2.cvtColor(attn_heatmap_colored, cv2.COLOR_RGB2BGR)
                overlayed_img = cv2.addWeighted(original_img_bgr, 0.2, attn_heatmap_bgr, 0.8, 0)

                ax.imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))
                ax.set_title(f'To Frame {tgt_frame_idx}', fontsize=10)

            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fpath = os.path.join(output_dir, "global", f"{title_prefix}_from_frame_{source_frame_idx}_grid.png")
        plt.savefig(fpath)
        plt.close(fig)
        print(f"\nSaved colorized attention grid to {fpath}")

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
    # Add a batch dimension, as the model expects (B, S, C, H, W)
    if images.dim() == 4:
        images = images.unsqueeze(0)

    # 4. Run Inference for Target Layers and Average the Attention Maps
    target_layers = [0, 22]
    print(f"\nStarting visualization for Global Layers {target_layers}...")
    
    normalized_maps_to_average = []
    B, S, _, H, W = images.shape
    patch_size = model.aggregator.patch_size
    patch_start_idx = model.aggregator.patch_start_idx
    P = (H // patch_size) * (W // patch_size) + patch_start_idx

    for target_layer in target_layers:
        print(f"--- Getting attention for Global Layer {target_layer:02d} ---")
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=dtype):
                _, raw_attn_map = model(
                    images,
                    visualize_attn_maps=True,
                    vis_target_layer=target_layer,
                    vis_source_frame=args.source_frame,
                )
                # Normalize this layer's map before adding it to the list for averaging
                normalized_map = normalize_attention_map(raw_attn_map, P, patch_size, patch_start_idx, H, W)
                normalized_maps_to_average.append(normalized_map)

    # Average the collected *normalized* attention maps
    avg_attn_map = np.mean(np.stack(normalized_maps_to_average, axis=0), axis=0)

    # 5. Save the Averaged Attention Maps
    save_attention_maps(
        normalized_avg_patch_map=avg_attn_map,
        B=B, S=S,
        patch_size=patch_size,
        output_dir=args.output_dir,
        images=images,
        source_frame_idx=args.source_frame,
        title_prefix=f"AVERAGE_L{target_layers[0]}_L{target_layers[1]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VGGT to visualize and save averaged attention maps for specified global layers.",
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