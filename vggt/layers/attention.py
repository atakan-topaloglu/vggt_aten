# vggt/layers/attention.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
from typing import Union, Tuple

from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, pos=None, visualize: bool = False, vis_source_cam_token_only: bool = False, source_frame_idx: int = 0, num_patches_per_frame: int = 0) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        use_fused_attn = self.fused_attn and not visualize

        attn_map = None
        if use_fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        elif vis_source_cam_token_only:
            # Memory-efficient path for visualizing specific attention
            # Query is only the camera token from the source frame
            # Key/Value are all tokens from all frames
            q_cam_token = q[:, :, source_frame_idx * num_patches_per_frame, :].unsqueeze(2) # Shape: (B, num_heads, 1, head_dim)

            attn = (q_cam_token * self.scale) @ k.transpose(-2, -1) # Shape: (B, num_heads, 1, S*P)
            attn = attn.softmax(dim=-1)
            attn_map = attn # Save for visualization
            attn = self.attn_drop(attn)
            x = attn @ v # Shape: (B, num_heads, 1, head_dim)
            # Since we only computed attention for one token, we need to update only that token's representation.
            # This is a simplification; a full implementation would scatter this back.
            # For visualization, we only need the attn_map, so the output 'x' is less critical.
            # We'll return a placeholder 'x' and the real attention map.
            # The aggregator will use the attention map and discard this 'x'.
            # A more robust implementation would involve scattering the result, but this is sufficient for visualization.
            return x, attn_map

        else:
            # Original, memory-intensive visualization path
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            if visualize:
                attn_map = attn
            attn = self.attn_drop(attn)
            x = attn @ v

        if not vis_source_cam_token_only:
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

        if visualize:
            if use_fused_attn:
                raise NotImplementedError("Cannot visualize with fused attention.")
            return x, attn_map
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, **kwargs) -> Tensor:
        if attn_bias is not None:
            warnings.warn("MemEffAttention with attn_bias does not support visualization.")
        # Pass kwargs to the parent's forward method
        return super().forward(x, pos=pos, visualize=False, **kwargs)