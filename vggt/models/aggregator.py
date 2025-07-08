# vggt/models/aggregator.py

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 # Import OpenCV
import warnings

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    # ... (keep __init__ and __build_patch_embed__ as they are) ...
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        self.patch_start_idx = 1 + num_register_tokens

        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        embed_dim=1024,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
    ):
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }
            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor, visualize_attn_maps: bool = False, visualize_output_dir: str = "attention_maps") -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        original_images_for_viz = images.clone() if visualize_attn_maps else None

        if visualize_attn_maps:
            if self.training:
                warnings.warn("Attention map visualization is disabled during training.")
                visualize_attn_maps = False
            else:
                print(f"Visualization of attention maps enabled. Saving to '{visualize_output_dir}'.")
                os.makedirs(os.path.join(visualize_output_dir, "frame"), exist_ok=True)
                os.makedirs(os.path.join(visualize_output_dir, "global"), exist_ok=True)

        images_normalized = (images - self._resnet_mean) / self._resnet_std
        images_normalized = images_normalized.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images_normalized)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
        if self.patch_start_idx > 0 and pos is not None:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape
        frame_idx, global_idx = 0, 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, visualize=visualize_attn_maps, output_dir=visualize_output_dir, images=original_images_for_viz
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, visualize=visualize_attn_maps, output_dir=visualize_output_dir, images=original_images_for_viz
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter, frame_intermediates, global_intermediates
        return output_list, self.patch_start_idx

    # --- MODIFICATION STARTS HERE ---
    def _save_attention_map(self, attn_map: torch.Tensor, attn_type: str, layer_idx: int, B: int, S: int, P: int, output_dir: str, images: torch.Tensor):
        attn_map_avg = attn_map.mean(dim=1).detach().cpu().numpy()
        
        patch_h = images.shape[-2] // self.patch_size
        patch_w = images.shape[-1] // self.patch_size
        num_patch_tokens = patch_h * patch_w

        if attn_type == 'frame':
            # Attention from camera token (idx 0) to patch tokens
            camera_token_attn = attn_map_avg[:, 0, self.patch_start_idx:]
            camera_token_attn = camera_token_attn.reshape(B * S, patch_h, patch_w)
            
            for i in range(B * S):
                batch_idx, frame_in_seq_idx = i // S, i % S
                if batch_idx > 0: continue

                original_img_tensor = images[batch_idx, frame_in_seq_idx]
                original_img_np = (original_img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

                attn_heatmap = camera_token_attn[i]
                attn_heatmap_resized = cv2.resize(attn_heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
                
                attn_heatmap_norm = (attn_heatmap_resized - attn_heatmap_resized.min()) / (attn_heatmap_resized.max() - attn_heatmap_resized.min() + 1e-8)
                attn_heatmap_colored = (plt.cm.viridis(attn_heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
                attn_heatmap_bgr = cv2.cvtColor(attn_heatmap_colored, cv2.COLOR_RGB2BGR)

                overlayed_img = cv2.addWeighted(original_img_bgr, 0.2, attn_heatmap_bgr, 0.8, 0)
                
                fpath = os.path.join(output_dir, "frame", f"layer_{layer_idx:02d}_frame_{frame_in_seq_idx:02d}_overlay.png")
                cv2.imwrite(fpath, overlayed_img)

        elif attn_type == 'global':
            # We want to see how the camera token of *every* frame (src)
            # attends to the patch tokens of *every* frame (tgt).
            # This will be visualized as an S x S grid.
            global_attn = attn_map_avg # Shape: (B, S*P, S*P)
            
            for b in range(B):
                if b > 0: continue # Only visualize for the first item in the batch

                # Create an S x S grid of subplots
                fig, axes = plt.subplots(S, S, figsize=(S * 4, S * 4), squeeze=False)
                fig.suptitle(f'Global Attention - Layer {layer_idx:02d}', fontsize=16)

                for src_frame_idx in range(S):
                    # The query token is the camera token of the source frame.
                    # Its index in the flattened S*P sequence is src_frame_idx * P.
                    query_token_idx = src_frame_idx * P
                    
                    for tgt_frame_idx in range(S):
                        # Calculate start and end indices for the patch tokens of the target frame
                        start = tgt_frame_idx * P + self.patch_start_idx
                        end = start + num_patch_tokens
                        
                        # Slice the attention weights from the source camera to the target patches
                        # global_attn has shape (B, S*P, S*P)
                        attn_slice = global_attn[b, query_token_idx, start:end].reshape(patch_h, patch_w)
                        
                        # Get the original image for overlaying
                        original_img_tensor = images[b, tgt_frame_idx]
                        original_img_np = (original_img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        original_img_bgr = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2BGR)

                        # Resize and color the heatmap
                        attn_heatmap_resized = cv2.resize(attn_slice, (original_img_np.shape[1], original_img_np.shape[0]))
                        attn_heatmap_norm = (attn_heatmap_resized - attn_heatmap_resized.min()) / (attn_heatmap_resized.max() - attn_heatmap_resized.min() + 1e-8)
                        attn_heatmap_colored = (plt.cm.viridis(attn_heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
                        attn_heatmap_bgr = cv2.cvtColor(attn_heatmap_colored, cv2.COLOR_RGB2BGR)
                        
                        # Overlay heatmap on the original image
                        overlayed_img = cv2.addWeighted(original_img_bgr, 0.2, attn_heatmap_bgr, 0.8, 0)
                        
                        # Plot on the corresponding subplot
                        ax = axes[src_frame_idx, tgt_frame_idx]
                        ax.imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))
                        
                        # Set titles for rows and columns
                        if tgt_frame_idx == 0:
                            ax.set_ylabel(f'From Frame {src_frame_idx}', fontsize=12)
                        if src_frame_idx == 0:
                            ax.set_title(f'To Frame {tgt_frame_idx}', fontsize=12)
                            
                        ax.set_xticks([])
                        ax.set_yticks([])

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                fpath = os.path.join(output_dir, "global", f"layer_{layer_idx:02d}_grid_overlay.png")
                plt.savefig(fpath)
                plt.close(fig)
    # --- MODIFICATION ENDS HERE ---

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, visualize=False, output_dir="attention_maps", images=None):
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)
        
        intermediates = []
        for _ in range(self.aa_block_size):
            block_input = tokens
            if visualize:
                tokens, attn_map = self.frame_blocks[frame_idx](block_input, pos=pos, visualize=True)
                self._save_attention_map(attn_map, 'frame', frame_idx, B, S, P, output_dir, images)
            else:
                tokens = self.frame_blocks[frame_idx](block_input, pos=pos)
            
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, visualize=False, output_dir="attention_maps", images=None):
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []
        for _ in range(self.aa_block_size):
            block_input = tokens
            if visualize:
                tokens, attn_map = self.global_blocks[global_idx](block_input, pos=pos, visualize=True)
                self._save_attention_map(attn_map, 'global', global_idx, B, S, P, output_dir, images)
            else:
                tokens = self.global_blocks[global_idx](block_input, pos=pos)

            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        return tokens, global_idx, intermediates

def slice_expand_and_flatten(token_tensor, B, S):
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined