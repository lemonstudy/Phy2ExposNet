import torch
from models.phy2_exposnet import Phy2ExposNet
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

class _UNetLikeFirstU_light(nn.Module):
    """A single U-Net (RadioUNet/RadioWNet style) returning one output tensor."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)

        if in_ch <= 3:
            self.layer00 = convrelu(in_ch, 6, 3, 1, 1)
            self.layer0  = convrelu(6, 24, 5, 2, 2)
            skip0_ch = 6
        else:
            self.layer00 = convrelu(in_ch, 8, 3, 1, 1)
            self.layer0  = convrelu(8, 24, 5, 2, 2)
            skip0_ch = 8

        self.layer1  = convrelu(24, 32, 5, 2, 2)
        self.layer10 = convrelu(32, 40, 5, 2, 1)
        self.layer2  = convrelu(40, 64, 5, 2, 2)
        self.layer20 = convrelu(64, 64, 3, 1, 1)
        self.layer3  = convrelu(64, 96, 5, 2, 2)
        self.layer4  = convrelu(96, 160, 5, 2, 2)
        self.layer5  = convrelu(160, 256, 5, 2, 2)

        self.conv_up5  = convreluT(256, 160, 4, 1)
        self.conv_up4  = convreluT(160 + 160, 96, 4, 1)
        self.conv_up3  = convreluT(96 + 96, 64, 4, 1)
        self.conv_up20 = convrelu(64 + 64, 64, 3, 1, 1)
        self.conv_up2  = convreluT(64 + 64, 40, 6, 2)
        self.conv_up10 = convrelu(40 + 40, 32, 5, 2, 1)
        self.conv_up1  = convreluT(32 + 32, 24, 6, 2)
        self.conv_up0  = convreluT(24 + 24, 16, 6, 2)

        self.conv_up00  = convrelu(16 + skip0_ch + in_ch, 16, 5, 2, 1)
        self.conv_up000 = convrelu(16 + in_ch, out_ch, 5, 2, 1)

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

class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for 2D feature maps: (B,C,H,W)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer).
    Attention is computed on channel dimension (per head), using spatial positions as tokens.
    """
    def __init__(self, dim: int, num_heads: int = 1, bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, heads, head_dim, HW)
        out = out.view(b, c, h, w)
        return self.proj(out)

class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network (Restormer)."""
    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66, bias: bool = True):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(self.project_in(x))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class RestormerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion_factor: float = 2.66, bias: bool = True):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim=dim, num_heads=num_heads, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)

class RestormerRefine(nn.Module):
    """A light Restormer for refinement: predicts delta map from pl0.

    Input:  (B, in_ch, H, W)  (typically in_ch=1)
    Output: (B, out_ch, H, W) (delta)
    """
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        dim: int = 48,
        num_blocks=(2, 2, 4),
        num_refinement_blocks: int = 2,
        heads=(1, 2, 4),
        ffn_expansion_factor: float = 2.66,
        bias: bool = True,
    ):
        super().__init__()
        assert len(num_blocks) == 3 and len(heads) == 3, "Use 3-stage encoder/decoder for this lite version."

        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1, bias=bias)

        self.enc1 = nn.Sequential(*[RestormerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])])
        self.down1 = Downsample(dim, dim * 2)

        self.enc2 = nn.Sequential(*[RestormerBlock(dim * 2, heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])])
        self.down2 = Downsample(dim * 2, dim * 4)

        self.latent = nn.Sequential(*[RestormerBlock(dim * 4, heads[2], ffn_expansion_factor, bias) for _ in range(num_blocks[2])])

        self.up2 = Upsample(dim * 4, dim * 2)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=bias)  # concat fuse
        self.dec2 = nn.Sequential(*[RestormerBlock(dim * 2, heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])])

        self.up1 = Upsample(dim * 2, dim)
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.dec1 = nn.Sequential(*[RestormerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])])

        self.refine = nn.Sequential(*[RestormerBlock(dim, heads[0], ffn_expansion_factor, bias) for _ in range(num_refinement_blocks)])

        self.out = nn.Conv2d(dim, out_ch, kernel_size=3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        e1 = self.enc1(x)
        x = self.down1(e1)

        e2 = self.enc2(x)
        x = self.down2(e2)

        x = self.latent(x)

        x = self.up2(x)
        x = self.reduce2(torch.cat([x, e2], dim=1))
        x = self.dec2(x)

        x = self.up1(x)
        x = self.reduce1(torch.cat([x, e1], dim=1))
        x = self.dec1(x)

        x = self.refine(x)
        return self.out(x)

class Stage1VIEModel(nn.Module):
    """
    Standalone physics-driven model.
    Input : (B,4,H,W) = [building, tx, Einc_re, Einc_im]
    Output: (B,2,H,W) = Etot (Re, Im)
    """
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.net = _UNetLikeFirstU(in_ch=in_ch, out_ch=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Stage2MSEModel(nn.Module):
    """
    Standalone data-driven model.
    Input : (B,4,H,W) = [building, tx, Einc_re, Einc_im]
    Output: (B,1,H,W) = predicted PL map
    """
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.net = RestormerRefine(
            in_ch=in_ch,
            out_ch=1,
            dim=16,
            num_blocks=(1, 1, 2),
            num_refinement_blocks=1,
            heads=(1, 1, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FullTwoStageModel(Phy2ExposNet):
    """Use the main model unchanged for the paper's full/no-physics ablations."""

    def forward_stage1(self, x4: torch.Tensor) -> torch.Tensor:
        return self.stage1(x4)

    def forward_stage2(self, pl0: torch.Tensor) -> torch.Tensor:
        return self.refine(pl0)


class LegacyTwoStageModel(nn.Module):
    """Original no-dropout, direct-output wrapper for historical checkpoints."""

    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.stage1 = _UNetLikeFirstU(in_ch=in_ch, out_ch=2)
        self.stage2 = RestormerRefine(in_ch=1, out_ch=1, dim=16,
                                    num_blocks=(1, 1, 2), num_refinement_blocks=1, heads=(1, 1, 2))

    def forward_stage1(self, x4: torch.Tensor) -> torch.Tensor:
        return self.stage1(x4)

    def forward_stage2(self, pl0: torch.Tensor) -> torch.Tensor:
        return self.stage2(pl0)
