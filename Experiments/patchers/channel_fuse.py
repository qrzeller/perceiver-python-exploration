"""Channel-fusing patch embedder

This module provides a lightweight patching + channel-fusing embedder that
turns local RGB patches into a compact token per spatial patch.

- Input expected as NHWC (B, H, W, C).
- Output is NHWC (B, H', W', out_dim).

Usage:
    from Experiments.patchers.channel_fuse import ChannelFusePatchEmbed

    patcher = ChannelFusePatchEmbed(in_ch=3, out_dim=1, patch_size=16, overlap=False)
    images = images.permute(0, 2, 3, 1).to(device)  # (B, H, W, C)
    tokens = patcher(images)  # (B, H', W', out_dim)

If you set `out_dim=1` the Perceiver should be instantiated with
`input_channels=1` (plus any fourier channels). For richer channel mixing set
`out_dim` to a larger value (e.g. 32 or 64) and pass that as `input_channels`.
"""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class ChannelFusePatchEmbed(nn.Module):
    """Fuse RGB channels and patchify the image into local tokens.

    Args:
        in_ch: number of input channels (3 for RGB)
        out_dim: output token dimensionality per patch (1 to collapse channels, >1 to keep features)
        patch_size: spatial patch size
        overlap: whether to use overlapping patches (stride = patch_size//2)
        use_bn: whether to use BatchNorm after conv
    """

    def __init__(self, in_ch: int = 3, out_dim: int = 1, patch_size: int = 16, overlap: bool = False, use_bn: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.stride = patch_size if not overlap else max(1, patch_size // 2)

        # Convolution that both patches (via kernel and stride) and fuses channels
        # Kernel size = patch_size, stride = stride. Output channels = out_dim.
        self.proj = nn.Conv2d(in_ch, out_dim, kernel_size=patch_size, stride=self.stride, bias=False)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_dim)
        else:
            self.bn = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x: (B, H, W, C) -> returns (B, H', W', out_dim)
        """
        assert x.ndim == 4, "Input must be NHWC"
        # NCHW for conv
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.proj(x)
        x = self.bn(x)
        x = self.act(x)
        # back to NHWC
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


# Small test helper (run from an interactive session)
if __name__ == "__main__":
    # smoke test
    B, H, W, C = 2, 224, 224, 3
    img = torch.randn(B, H, W, C)
    patcher = ChannelFusePatchEmbed(in_ch=3, out_dim=1, patch_size=16, overlap=False)
    out = patcher(img)
    print('in:', img.shape, 'out:', out.shape)
