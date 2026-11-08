import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

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
            self.layer0 = convrelu(6, 60, 5, 2, 2)
            skip0_ch = 6
        else:
            self.layer00 = convrelu(in_ch, 10, 3, 1, 1)
            self.layer0 = convrelu(10, 60, 5, 2, 2)
            skip0_ch = 10

        self.layer1 = convrelu(60, 80, 5, 2, 2)
        self.layer10 = convrelu(80, 100, 5, 2, 1)
        self.layer2 = convrelu(100, 128, 5, 2, 2)
        self.layer20 = convrelu(128, 128, 3, 1, 1)
        self.layer3 = convrelu(128, 256, 5, 2, 2)
        self.layer4 = convrelu(256, 512, 5, 2, 2)
        self.layer5 = convrelu(512, 1024, 5, 2, 2)

        self.conv_up5 = convreluT(1024, 512, 4, 1)
        self.conv_up4 = convreluT(512 + 512, 256, 4, 1)
        self.conv_up3 = convreluT(256 + 256, 128, 4, 1)
        self.conv_up20 = convrelu(128 + 128, 128, 3, 1, 1)
        self.conv_up2 = convreluT(128 + 128, 100, 6, 2)
        self.conv_up10 = convrelu(100 + 100, 80, 5, 2, 1)
        self.conv_up1 = convreluT(80 + 80, 60, 6, 2)
        self.conv_up0 = convreluT(60 + 60, 20, 6, 2)

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

class PEFNet(nn.Module):
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.stage1 = _UNetLikeFirstU(in_ch=in_ch, out_ch=2)  # Etot (Re,Im)
        self.stage2 = _UNetLikeFirstU(in_ch=1, out_ch=1)      # delta PL

    def forward(self, x4: torch.Tensor) -> torch.Tensor:
        """Stage-1 only: x4 -> etot_ri"""
        return self.stage1(x4)

    def refine(self, pl0: torch.Tensor) -> torch.Tensor:
        """Stage-2 only: pl0 -> pl_ref"""
        delta = self.stage2(pl0)
        return pl0 + delta
