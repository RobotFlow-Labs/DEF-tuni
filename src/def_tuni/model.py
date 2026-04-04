"""TUNI: Real-time RGB-T Semantic Segmentation Model.

Self-contained implementation — no mmcv dependency.
Reference: repositories/TUNI/backbone_model/TUNI.py + model1.py + decoder/MLP.py
"""
from __future__ import annotations

import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# mmcv replacements
# ---------------------------------------------------------------------------

def build_norm_layer(cfg: dict, num_features: int):
    """Minimal replacement for mmcv.cnn.build_norm_layer."""
    norm_type = cfg.get("type", "BN")
    if norm_type == "BN":
        layer = nn.BatchNorm2d(num_features)
    elif norm_type == "LN":
        layer = nn.LayerNorm(num_features)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    return norm_type, layer


class DropPath(nn.Module):
    """Stochastic depth — drop entire residual branch."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).add_(keep).floor_()
        return x.div(keep) * mask


def build_dropout(cfg: dict | None):
    if cfg is None:
        return nn.Identity()
    if cfg.get("type") == "DropPath":
        return DropPath(cfg.get("drop_prob", 0.0))
    return nn.Identity()


# ---------------------------------------------------------------------------
# Backbone layers
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1,
                             groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        return self.fc2(x)


class LocalAttentionRGBT(nn.Module):
    def __init__(self, dim: int, ratio: int = 8):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim // 2)
        self.linear2 = nn.Linear(dim // 2, dim // 2)
        self.conv1 = nn.Conv2d(dim // 2, dim // 2, 7, padding=3, groups=dim // 2)
        self.conv2 = nn.Conv2d(dim // 2, dim // 2, 7, padding=3, groups=dim // 2)
        self.fc_channel = nn.Sequential(
            nn.Linear(dim, dim // ratio, bias=False),
            nn.GELU(),
            nn.Linear(dim // ratio, dim, bias=False),
        )
        self.linear3 = nn.Linear(dim, dim // 2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)

    def forward(self, rgb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        rgb = self.linear1(rgb).permute(0, 3, 1, 2)
        t = self.linear2(t).permute(0, 3, 1, 2)
        co = self.conv1(rgb * t)
        di = self.conv2(torch.abs(rgb - t))
        co_di = torch.cat([co, di], 1).permute(0, 2, 3, 1)

        attention_map = torch.mean(co_di.permute(0, 3, 1, 2), dim=1, keepdim=True).permute(0, 2, 3, 1)
        dot_product = torch.sum(attention_map * co_di, dim=[1, 2], keepdim=True)
        norm_1 = torch.norm(attention_map, p=2, dim=[1, 2], keepdim=True)
        norm_2 = torch.norm(co_di, p=2, dim=[1, 2], keepdim=True)
        cos_sim = dot_product / (norm_1 * norm_2 + 1e-6)
        attention_c = self.fc_channel(cos_sim).sigmoid()
        return self.linear3(co_di * attention_c)


class Attention(nn.Module):
    def __init__(self, dim: int, num_head: int = 8, window: int = 7,
                 drop_depth: bool = False):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.local_rr1 = nn.Linear(dim, dim)
        self.local_rr2 = nn.Linear(dim, dim)
        self.local_rr3 = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.local_rx = LocalAttentionRGBT(dim)

        self.proj = nn.Linear(dim // 2 * 3, dim)
        if not drop_depth:
            self.proj_e = nn.Linear(dim // 2 * 3, dim // 2)
        if window != 0:
            self.kv_rx = nn.Linear(dim, dim)
            self.q_rx = nn.Linear(dim // 2 * 3, dim // 2)
            self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
            self.proj = nn.Linear(dim * 2, dim)
            if not drop_depth:
                self.proj_e = nn.Linear(dim * 2, dim // 2)

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim // 2, eps=1e-6, data_format="channels_last")
        self.drop_depth = drop_depth

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        B, H, W, C = x.size()
        x = self.norm(x)
        x_e = self.norm_e(x_e)

        # local RGB-RGB attention
        local_rr1 = self.local_rr1(x)
        local_rr2 = self.local_rr2(x).permute(0, 3, 1, 2)
        local_rr2 = self.local_rr3(self.conv(local_rr2).permute(0, 2, 3, 1))
        local_rr = local_rr1 * local_rr2

        # local RGB-T attention
        local_rx = self.local_rx(x, x_e)

        if self.window != 0:
            # global RGB-T attention
            kv_rx = self.kv_rx(x)
            kv_rx = kv_rx.reshape(B, H * W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
            k_rx, v_rx = kv_rx.unbind(0)
            rx = torch.cat([x, x_e], dim=3)
            rx_pool = self.pool(rx.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            q_rx = self.q_rx(rx_pool)
            q_rx = q_rx.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)
            global_rx = (q_rx * (C // self.num_head // 2) ** -0.5) @ k_rx.transpose(-2, -1)
            global_rx = global_rx.softmax(dim=-1)
            global_rx = (global_rx @ v_rx).reshape(
                B, self.num_head, self.window, self.window, C // self.num_head // 2
            ).permute(0, 1, 4, 2, 3).reshape(B, C // 2, self.window, self.window)
            global_rx = F.interpolate(global_rx, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            x = torch.cat([local_rr, local_rx, global_rx], dim=3)
        else:
            x = torch.cat([local_rr, local_rx], dim=3)

        if not self.drop_depth:
            x_e = self.proj_e(x)
        else:
            x_e = local_rx

        x = self.proj(x)
        return x, x_e


class Block(nn.Module):
    def __init__(self, index: int, dim: int, num_head: int, mlp_ratio: float = 4.,
                 block_index: int = 0, last_block_index: int = 50, window: int = 7,
                 dropout_layer=None, drop_depth: bool = False):
        super().__init__()
        self.index = index
        layer_scale_init_value = 1e-6
        if block_index > last_block_index:
            window = 0
        self.attn = Attention(dim, num_head, window=window, drop_depth=drop_depth)
        self.mlp = MLP(dim, int(mlp_ratio))
        self.dropout_layer = build_dropout(dropout_layer)

        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

        if not drop_depth:
            self.layer_scale_1_e = nn.Parameter(layer_scale_init_value * torch.ones(dim // 2))
            self.layer_scale_2_e = nn.Parameter(layer_scale_init_value * torch.ones(dim // 2))
            self.mlp_e2 = MLP(dim // 2, int(mlp_ratio))

        self.drop_depth = drop_depth

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        res_x, res_e = x, x_e
        x, x_e = self.attn(x, x_e)

        x = res_x + self.dropout_layer(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x)
        x = x + self.dropout_layer(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x))

        if not self.drop_depth:
            x_e = res_e + self.dropout_layer(self.layer_scale_1_e.unsqueeze(0).unsqueeze(0) * x_e)
            x_e = x_e + self.dropout_layer(self.layer_scale_2_e.unsqueeze(0).unsqueeze(0) * self.mlp_e2(x_e))

        return x, x_e


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------

class TUNIBackbone(nn.Module):
    def __init__(self, dims=(48, 96, 192, 384), depths=(2, 2, 4, 2),
                 mlp_ratios=(8, 8, 4, 4), num_heads=(1, 2, 4, 8),
                 windows=(0, 7, 7, 7), last_block=(50, 50, 50, 50),
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.depths = depths
        norm_cfg = dict(type='BN', requires_grad=True)

        # RGB stem
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)

        # Thermal stem (single channel input, half dims)
        self.downsample_layers_e = nn.ModuleList()
        stem_e = nn.Sequential(
            nn.Conv2d(1, dims[0] // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 4),
            nn.GELU(),
            nn.Conv2d(dims[0] // 4, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0] // 2),
        )
        self.downsample_layers_e.append(stem_e)

        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                build_norm_layer(norm_cfg, dims[i])[1],
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

            downsample_layer_e = nn.Sequential(
                build_norm_layer(norm_cfg, dims[i] // 2)[1],
                nn.Conv2d(dims[i] // 2, dims[i + 1] // 2, kernel_size=3, stride=2, padding=1),
            )
            self.downsample_layers_e.append(downsample_layer_e)

        # Build stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(
                    index=cur + j,
                    dim=dims[i],
                    window=windows[i],
                    dropout_layer=dict(type='DropPath', drop_prob=dp_rates[cur + j]),
                    num_head=num_heads[i],
                    block_index=depths[i] - j,
                    last_block_index=last_block[i],
                    mlp_ratio=mlp_ratios[i],
                    drop_depth=((i == 3) and (j == depths[i] - 1))
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict_ema" in ckpt:
            sd = ckpt["state_dict_ema"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        clean = OrderedDict()
        for k, v in sd.items():
            if k.startswith("Backbone."):
                k = k[9:]
            if k.startswith("module."):
                k = k[7:]
            clean[k] = v

        missing, unexpected = self.load_state_dict(clean, strict=False)
        if missing:
            print(f"[TUNI] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[TUNI] Unexpected keys: {len(unexpected)}")

    def forward(self, x: torch.Tensor, x_e: torch.Tensor | None = None):
        if x_e is None:
            x_e = x
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(x_e.shape) == 3:
            x_e = x_e.unsqueeze(0)

        # Extract single thermal channel
        x_e = x_e[:, 0:1, :, :]

        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x_e = self.downsample_layers_e[i](x_e)

            x = x.permute(0, 2, 3, 1)
            x_e = x_e.permute(0, 2, 3, 1)

            for blk in self.stages[i]:
                x, x_e = blk(x, x_e)

            x = x.permute(0, 3, 1, 2)
            x_e = x_e.permute(0, 3, 1, 2)
            outs.append(x)
        return outs


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderMLP(nn.Module):
    def __init__(self, in_channels=(48, 96, 192, 384), embed_dim: int = 256,
                 num_classes: int = 9, dropout_ratio: float = 0.1):
        super().__init__()
        self.num_classes = num_classes

        self.linear_c4 = nn.Linear(in_channels[3], embed_dim)
        self.linear_c3 = nn.Linear(in_channels[2], embed_dim)
        self.linear_c2 = nn.Linear(in_channels[1], embed_dim)
        self.linear_c1 = nn.Linear(in_channels[0], embed_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def _project(self, feat: torch.Tensor, linear: nn.Linear, target_size) -> torch.Tensor:
        n, c, h, w = feat.shape
        x = feat.flatten(2).transpose(1, 2)  # (n, h*w, c)
        x = linear(x)  # (n, h*w, embed)
        x = x.permute(0, 2, 1).reshape(n, -1, h, w)
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        target_size = c1.shape[2:]

        _c4 = self._project(c4, self.linear_c4, target_size)
        _c3 = self._project(c3, self.linear_c3, target_size)
        _c2 = self._project(c2, self.linear_c2, target_size)
        _c1 = self._project(c1, self.linear_c1, target_size)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        return self.linear_pred(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

# Backbone configurations
BACKBONE_CONFIGS = {
    "tiny": dict(dims=(32, 64, 128, 256), depths=(3, 4, 4, 6),
                 mlp_ratios=(8, 8, 4, 4), num_heads=(1, 2, 4, 8),
                 windows=(0, 7, 7, 7)),
    "384_2242": dict(dims=(48, 96, 192, 384), depths=(2, 2, 4, 2),
                     mlp_ratios=(8, 8, 4, 4), num_heads=(1, 2, 4, 8),
                     windows=(0, 7, 7, 7)),
    "512_2242": dict(dims=(64, 128, 256, 512), depths=(2, 2, 4, 2),
                     mlp_ratios=(8, 8, 4, 4), num_heads=(1, 2, 4, 8),
                     windows=(0, 7, 7, 7)),
    "320_2262": dict(dims=(40, 80, 160, 320), depths=(2, 2, 6, 2),
                     mlp_ratios=(8, 8, 4, 4), num_heads=(1, 2, 4, 8),
                     windows=(0, 7, 7, 7)),
}


class TUNIModel(nn.Module):
    """Full TUNI model: backbone encoder + MLP decoder."""

    def __init__(self, variant: str = "384_2242", n_classes: int = 9,
                 drop_path_rate: float = 0.1, pretrained: str | None = None):
        super().__init__()
        cfg = BACKBONE_CONFIGS[variant]
        self.encoder = TUNIBackbone(drop_path_rate=drop_path_rate, **cfg)
        self.decoder = DecoderMLP(
            in_channels=cfg["dims"],
            embed_dim=256,
            num_classes=n_classes,
        )
        if pretrained:
            self.encoder.load_pretrained(pretrained)

    def load_checkpoint(self, path: str):
        """Load full model checkpoint (encoder + decoder) from reference format.

        Reference weights use: encoder.enc.* → our encoder.*
                               decoder.* → decoder.* (compatible via .proj rename)
        """
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]

        remapped = OrderedDict()
        for k, v in sd.items():
            # Reference: encoder.enc.X → our: encoder.X
            if k.startswith("encoder.enc."):
                remapped["encoder." + k[12:]] = v
            # Reference decoder.linear_cN.proj.* → our decoder.linear_cN.*
            elif k.startswith("decoder.") and ".proj." in k:
                new_k = k.replace(".proj.", ".")
                remapped[new_k] = v
            else:
                remapped[k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"[TUNI] Loaded {loaded}/{len(sd)} keys (missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            print(f"  Missing sample: {missing[:3]}")

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor | None = None):
        if thermal is None:
            thermal = rgb
        features = self.encoder(rgb, thermal)
        sem = self.decoder(features)
        return F.interpolate(sem, scale_factor=4, mode='bilinear', align_corners=False)

    @torch.no_grad()
    def predict(self, rgb: torch.Tensor, thermal: torch.Tensor | None = None):
        self.eval()
        logits = self.forward(rgb, thermal)
        return logits.argmax(dim=1)
