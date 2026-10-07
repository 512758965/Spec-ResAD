"""Residual features, hypersphere constraint, and a coupling flow."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, wide_resnet50_2


def imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = x.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (x - mean) / std


def load_checkpoint(path: str | Path) -> dict:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    return obj


def residual_feature(feat: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
    """Subtract the nearest reference vector at every location."""
    b, c, h, w = feat.shape
    query = feat.reshape(b, c, h * w).permute(0, 2, 1).reshape(-1, c)
    best_i = torch.zeros(query.shape[0], dtype=torch.long, device=feat.device)
    query_step = 256
    pool_step = 4096
    for qs in range(0, query.shape[0], query_step):
        qe = min(qs + query_step, query.shape[0])
        qq = query[qs:qe]
        local_d = torch.full((qq.shape[0],), float("inf"), device=feat.device)
        local_i = torch.zeros(qq.shape[0], dtype=torch.long, device=feat.device)
        for ps in range(0, pool.shape[0], pool_step):
            pe = min(ps + pool_step, pool.shape[0])
            dist = torch.cdist(qq, pool[ps:pe])
            d, index = dist.min(dim=1)
            take = d < local_d
            local_d = torch.where(take, d, local_d)
            local_i = torch.where(take, index + ps, local_i)
        best_i[qs:qe] = local_i
    nearest = pool[best_i]
    residual = (query - nearest).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
    return residual


class AffineCoupling(nn.Module):
    def __init__(self, channels: int, hidden: int = 128, swap: bool = False):
        super().__init__()
        self.swap = swap
        self.c1 = channels // 2
        c2 = channels - self.c1
        self.net = nn.Sequential(
            nn.Conv2d(self.c1, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c2 * 2, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.swap:
            x = x.flip(1)
        x1, x2 = x[:, : self.c1], x[:, self.c1 :]
        scale, shift = self.net(x1).chunk(2, dim=1)
        scale = 0.5 * torch.tanh(scale)
        y2 = x2 * torch.exp(scale) + shift
        y = torch.cat([x1, y2], dim=1)
        if self.swap:
            y = y.flip(1)
        return y, scale.sum(dim=1)


class CouplingFlow(nn.Module):
    def __init__(self, channels: int, n_layers: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [AffineCoupling(channels, swap=(i % 2 == 1)) for i in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = x.new_zeros(x.shape[0], x.shape[2], x.shape[3])
        for layer in self.layers:
            x, piece = layer(x)
            logdet = logdet + piece
        nll = 0.5 * x.pow(2).sum(dim=1) - logdet
        return x, nll


class ConvBnReLU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(x + y)


class Bottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 4, 32)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(x + y)


def make_constraintor(arch: str, channels: int) -> nn.Module:
    if arch == "convbnrelu":
        return ConvBnReLU(channels)
    if arch == "basicblock":
        return BasicBlock(channels)
    if arch == "bottleneck":
        return Bottleneck(channels)
    raise ValueError(f"Unknown constraintor_arch: {arch}")


class WideResNetFeatures(nn.Module):
    def __init__(self, weights_path: str | Path):
        super().__init__()
        net = wide_resnet50_2(weights=None)
        net.load_state_dict(load_checkpoint(weights_path))
        self.net = net
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        net = self.net
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        f1 = net.layer1(x)
        f2 = net.layer2(f1)
        f3 = net.layer3(f2)
        return [f1, f2, f3]


class ViTFeatures(nn.Module):
    """Intermediate ViT-B/16 blocks."""

    def __init__(self, weights_path: str | Path, blocks: tuple[int, ...] = (4, 8, 12)):
        super().__init__()
        net = vit_b_16(weights=None)
        net.load_state_dict(load_checkpoint(weights_path))
        self.net = net
        self.blocks = tuple(b - 1 for b in blocks)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        net = self.net
        n = x.shape[0]
        tokens = net._process_input(x)
        tokens = torch.cat([net.class_token.expand(n, -1, -1), tokens], dim=1)
        tokens = tokens + net.encoder.pos_embedding
        tokens = net.encoder.dropout(tokens)
        feats = []
        for index, layer in enumerate(net.encoder.layers):
            tokens = layer(tokens)
            if index in self.blocks:
                patch = tokens[:, 1:, :]
                side = int(patch.shape[1] ** 0.5)
                feats.append(patch.transpose(1, 2).reshape(n, -1, side, side))
        return feats


class SpecResAD(nn.Module):
    def __init__(self, cfg: dict, weights_path: str):
        super().__init__()
        self.cfg = cfg
        self.use_residual = bool(cfg.get("use_residual", True))
        self.use_constraintor = bool(cfg.get("use_constraintor", True))
        self.abnormal_invariant = bool(cfg.get("abnormal_invariant", True))
        backbone = cfg.get("backbone", "wide_resnet50")
        if backbone == "wide_resnet50":
            self.extractor = WideResNetFeatures(weights_path)
        elif backbone == "vit_b_16":
            blocks = tuple(cfg.get("vit_blocks", [4, 8, 12]))
            self.extractor = ViTFeatures(weights_path, blocks=blocks)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.extractor.eval()
        with torch.no_grad():
            size = int(cfg["image_size"])
            dummy = self.extractor(imagenet_norm(torch.zeros(1, 3, size, size)))
        channels = [feat.shape[1] for feat in dummy]
        arch = cfg.get("constraintor_arch", "convbnrelu")
        n_layers = int(cfg.get("flow_layers", 8))
        self.constraintors = nn.ModuleList([make_constraintor(arch, c) for c in channels])
        self.flows = nn.ModuleList([CouplingFlow(c, n_layers) for c in channels])

    def train(self, mode: bool = True):
        super().train(mode)
        self.extractor.eval()
        return self

    def extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.extractor(imagenet_norm(x))

    def _constrain(self, residual: torch.Tensor, level: int) -> torch.Tensor:
        if self.use_constraintor:
            return self.constraintors[level](residual)
        return residual

    def forward_train(self, images: torch.Tensor, refs: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        """refs: (B, K, 3, H, W), an independent reference set for each image."""
        b, k, _, h, w = refs.shape
        with torch.no_grad():
            query = self.extract(images)
            reference = self.extract(refs.reshape(b * k, 3, h, w))
        occ_terms = []
        ml_terms = []
        bg_terms = []
        for level, (qf, rf) in enumerate(zip(query, reference)):
            residual = self._per_image_residual(qf, rf, k)
            constrained = self._constrain(residual, level)
            _, nll = self.flows[level](constrained)
            occ_terms.append(self._occ(residual, constrained, masks))
            ml_terms.append(nll.mean())
            bg_terms.append(self._bg(nll, masks))
        occ = torch.stack(occ_terms).mean()
        ml = torch.stack(ml_terms).mean()
        bg = torch.stack(bg_terms).mean()
        if not self.use_constraintor:
            occ = occ * 0.0
        total = occ + ml + float(self.cfg.get("lambda_bg", 0.1)) * bg
        return {"loss": total, "occ": occ.detach(), "ml": ml.detach(), "bg": bg.detach()}

    def _per_image_residual(self, query: torch.Tensor, reference: torch.Tensor, k: int) -> torch.Tensor:
        if not self.use_residual:
            return query
        b, c, h, w = query.shape
        reference = reference.view(b, k, c, h, w)
        pieces = []
        for i in range(b):
            pool = reference[i].permute(0, 2, 3, 1).reshape(-1, c)
            pieces.append(residual_feature(query[i : i + 1], pool))
        return torch.cat(pieces, dim=0)

    def _occ(self, residual: torch.Tensor, constrained: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        y = F.interpolate(masks.float(), size=residual.shape[-2:], mode="nearest")
        y = (y > 0.5).float()
        norm = constrained.pow(2).sum(dim=1, keepdim=True)
        change = (constrained - residual).pow(2).sum(dim=1, keepdim=True)
        if self.abnormal_invariant:
            per = (1.0 - y) * norm + (1.0 + y) * change
        else:
            per = (1.0 - y) * norm
        return per.mean()

    def _bg(self, nll: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        logp = -nll
        y = F.interpolate(masks.float(), size=nll.shape[-2:], mode="nearest")[:, 0] > 0.5
        boundary = float(self.cfg.get("boundary", 0.0))
        margin = float(self.cfg.get("margin_tau", 1.0))
        loss = nll.new_zeros(())
        if (~y).any():
            loss = loss + F.relu(boundary - logp[~y]).mean()
        if y.any():
            loss = loss + F.relu(logp[y] - (boundary - margin)).mean()
        return loss

    @torch.no_grad()
    def score(self, images: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        """refs: (K, 3, H, W), shared by every image in the batch. Returns NLL maps."""
        was_training = self.training
        self.eval()
        query = self.extract(images)
        reference = self.extract(refs)
        maps = []
        for level, (qf, rf) in enumerate(zip(query, reference)):
            pool = rf.permute(0, 2, 3, 1).reshape(-1, rf.shape[1])
            residual = residual_feature(qf, pool) if self.use_residual else qf
            constrained = self._constrain(residual, level)
            _, nll = self.flows[level](constrained)
            maps.append(F.interpolate(nll[:, None], size=images.shape[-2:], mode="bilinear", align_corners=False))
        anomaly = torch.stack(maps, dim=0).mean(dim=0)
        if was_training:
            self.train()
        return anomaly
