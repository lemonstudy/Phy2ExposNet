import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),

        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False) 

    )

def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.LeakyReLU(inplace=True)

    )

class _UNetLikeFirstU(nn.Module):
    """A single U-Net (RadioUNet/RadioWNet style) returning one output tensor."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)

        if in_ch <= 3:
            self.layer00 = convrelu(in_ch, 6, 3, 1, 1)
            self.layer0 = convrelu(6, 40, 5, 2, 2)
            skip0_ch = 6
        else:
            self.layer00 = convrelu(in_ch, 10, 3, 1, 1)
            self.layer0 = convrelu(10, 40, 5, 2, 2)
            skip0_ch = 10

        self.layer1 = convrelu(40, 50, 5, 2, 2)
        self.layer10 = convrelu(50, 60, 5, 2, 1)
        self.layer2 = convrelu(60, 100, 5, 2, 2)
        self.layer20 = convrelu(100, 100, 3, 1, 1)
        self.layer3 = convrelu(100, 150, 5, 2, 2)
        self.layer4 = convrelu(150, 300, 5, 2, 2)
        self.layer5 = convrelu(300, 500, 5, 2, 2)

        self.conv_up5 = convreluT(500, 300, 4, 1)
        self.conv_up4 = convreluT(300 + 300, 150, 4, 1)
        self.conv_up3 = convreluT(150 + 150, 100, 4, 1)
        self.conv_up20 = convrelu(100 + 100, 100, 3, 1, 1)
        self.conv_up2 = convreluT(100 + 100, 60, 6, 2)
        self.conv_up10 = convrelu(60 + 60, 50, 5, 2, 1)
        self.conv_up1 = convreluT(50 + 50, 40, 6, 2)
        self.conv_up0 = convreluT(40 + 40, 20, 6, 2)

        self.conv_up00 = convrelu(20 + skip0_ch + in_ch, 20, 5, 2, 1)
        self.conv_up000 = convrelu(20 + in_ch, out_ch, 5, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x0 = x[:, 0:self.in_ch, :, :]

        l00 = self.layer00(x0)
        l0 = self.layer0(l00)
        l1 = self.layer1(l0)
        l10 = self.layer10(l1)
        l2 = self.layer2(l10)
        l20 = self.layer20(l2)
        l3 = self.layer3(l20)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)

        u4 = self.conv_up5(l5)
        u4 = torch.cat([u4, l4], dim=1)
        u3 = self.conv_up4(u4)
        u3 = torch.cat([u3, l3], dim=1)
        u20 = self.conv_up3(u3)
        u20 = torch.cat([u20, l20], dim=1)
        u2 = self.conv_up20(u20)
        u2 = torch.cat([u2, l2], dim=1)
        u10 = self.conv_up2(u2)
        u10 = torch.cat([u10, l10], dim=1)
        u1 = self.conv_up10(u10)
        u1 = torch.cat([u1, l1], dim=1)
        u0 = self.conv_up1(u1)
        u0 = torch.cat([u0, l0], dim=1)
        u00 = self.conv_up0(u0)
        u00 = torch.cat([u00, l00], dim=1)
        u00 = torch.cat([u00, x0], dim=1)
        u000 = self.conv_up00(u00)
        u000 = torch.cat([u000, x0], dim=1)
        out = self.conv_up000(u000)
        return out

class ResBlockGN(nn.Module):
    """A simple residual block at fixed resolution.

    Uses GroupNorm for stability with small batch sizes (e.g., batch=16).
    """
    def __init__(self, ch: int, gn_groups: int = 8):
        super().__init__()
        g = min(gn_groups, ch)

        while ch % g != 0 and g > 1:
            g -= 1
        self.norm1 = nn.GroupNorm(g, ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True)
        self.norm2 = nn.GroupNorm(g, ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.act(self.norm1(x)))
        y = self.conv2(self.act(self.norm2(y)))
        return x + y

class ResNetRefine(nn.Module):
    """Fixed-resolution ResNet refine: predicts delta map from pl0.

    Input:  (B, in_ch, H, W)
    Output: (B, out_ch, H, W)  (delta)
    """
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 96,
        n_blocks: int = 10,
        gn_groups: int = 8,
    ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=True)
        self.blocks = nn.Sequential(*[ResBlockGN(base_ch, gn_groups=gn_groups) for _ in range(n_blocks)])
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1, bias=True)

        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.blocks(x)
        return self.out_conv(x)

class ResNet(nn.Module):
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.stage1 = _UNetLikeFirstU(in_ch=in_ch, out_ch=2)  # Etot (Re,Im)
        self.stage2 = ResNetRefine(in_ch=1, out_ch=1, base_ch=96, n_blocks=10)

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        """Stage-1 only: x4 -> etot_ri"""
        return self.stage1(x4)

    def refine(self, pl0: torch.Tensor) -> torch.Tensor:
        """Stage-2 only: pl0 -> pl_ref"""
        delta = self.stage2(pl0)
        return pl0 + delta
