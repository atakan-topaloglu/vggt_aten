# vggt/layers/visual_block.py

from typing import Callable
import torch
from torch import nn, Tensor

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp

class VisualBlock(nn.Module):
    """
    A modified version of the standard Block that returns the attention map
    for visualization purposes. Used only by the Aggregator.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, pos=None) -> tuple[Tensor, Tensor]:
        # CORRECTED LOGIC:
        # 1. Call self.attn and unpack the tuple immediately.
        attn_output, attn_map = self.attn(self.norm1(x), pos=pos)
        
        # 2. Use *only* the tensor part (attn_output) for the first residual connection.
        x = x + self.drop_path1(self.ls1(attn_output))
        
        # 3. The FFN part remains the same.
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        # 4. Return the final tensor and the saved attention map.
        return x, attn_map