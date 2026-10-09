"""
Model definitions for Beta evidence, EDL, BNN, SWAG, and MaxSoftmax.
All methods share the same encoder backbone and differ only in the head/output.
"""

from collections import deque
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class EvidentialHead(nn.Module):
    """Beta evidence head that outputs positive and negative evidence."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 2 * num_classes)
        self.num_classes = num_classes
        self.eps = 1e-4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.fc(x)
        evidence = F.softplus(out)
        pos_evidence, neg_evidence = evidence.chunk(2, dim=-1)
        alpha = pos_evidence + 1.0 + self.eps
        beta = neg_evidence + 1.0 + self.eps
        return alpha, beta


class DirichletHead(nn.Module):
    """Dirichlet evidential head for standard EDL."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.eps = 1e-4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        evidence = F.softplus(self.fc(x))
        alpha = evidence + 1.0 + self.eps
        return evidence, alpha


class ClassifierHead(nn.Module):
    """Classifier head used by BNN/MC Dropout and SWAG."""

    def __init__(self, in_features: int, num_classes: int, dropout_p: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class CNN1DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + x, inplace=True)


class ResNet1DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResidualBlock1D(64), ResidualBlock1D(64))
        self.transition1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm1d(128),
        )
        self.layer2 = nn.Sequential(ResidualBlock1D(128), ResidualBlock1D(128))
        self.transition2 = nn.Sequential(
            nn.Conv1d(128, hidden_dim, kernel_size=1, stride=2),
            nn.BatchNorm1d(hidden_dim),
        )
        self.layer3 = nn.Sequential(ResidualBlock1D(hidden_dim), ResidualBlock1D(hidden_dim))
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        return x


class MultiScaleResNet1DEncoder(nn.Module):
    """1D encoder with early multi-scale convolution branches."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        base_branch_dim = hidden_dim // 3
        branch_dims = [base_branch_dim, base_branch_dim, hidden_dim - 2 * base_branch_dim]
        kernels = (3, 7, 15)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_dim, kernel_size=kernel, stride=2, padding=kernel // 2),
                    nn.BatchNorm1d(branch_dim),
                    nn.ReLU(inplace=True),
                )
                for branch_dim, kernel in zip(branch_dims, kernels)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.residual = nn.Sequential(
            ResidualBlock1D(hidden_dim),
            ResidualBlock1D(hidden_dim),
            ResidualBlock1D(hidden_dim),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.fusion(x)
        return self.residual(x)


class CNN2DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.block(x) + x, inplace=True)


class ResNet2DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResidualBlock2D(64), ResidualBlock2D(64))
        self.transition1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2),
            nn.BatchNorm2d(128),
        )
        self.layer2 = nn.Sequential(ResidualBlock2D(128), ResidualBlock2D(128))
        self.transition2 = nn.Sequential(
            nn.Conv2d(128, hidden_dim, kernel_size=1, stride=2),
            nn.BatchNorm2d(hidden_dim),
        )
        self.layer3 = nn.Sequential(ResidualBlock2D(hidden_dim), ResidualBlock2D(hidden_dim))
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        return x


class StaticFeatureExtractor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        in_ch = 1
        if config.encoder_type == "multiscale_resnet1d":
            if config.input_dim != "1d":
                raise ValueError("multiscale_resnet1d only supports input_dim='1d'")
            self.encoder = MultiScaleResNet1DEncoder(in_ch, config.hidden_dim)
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif config.input_dim == "1d":
            self.encoder = ResNet1DEncoder(in_ch, config.hidden_dim) if config.encoder_type == "resnet" else CNN1DEncoder(in_ch, config.hidden_dim)
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.encoder = ResNet2DEncoder(in_ch, config.hidden_dim) if config.encoder_type == "resnet" else CNN2DEncoder(in_ch, config.hidden_dim)
            self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = config.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.pool(feat)
        return feat.flatten(1)


class SequentialFeatureExtractor(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        in_ch = 1
        if config.encoder_type == "multiscale_resnet1d":
            if config.input_dim != "1d":
                raise ValueError("multiscale_resnet1d only supports input_dim='1d'")
            self.encoder = MultiScaleResNet1DEncoder(in_ch, config.hidden_dim)
            self.frame_pool = nn.AdaptiveAvgPool1d(1)
        elif config.input_dim == "1d":
            self.encoder = ResNet1DEncoder(in_ch, config.hidden_dim) if config.encoder_type == "resnet" else CNN1DEncoder(in_ch, config.hidden_dim)
            self.frame_pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.encoder = ResNet2DEncoder(in_ch, config.hidden_dim) if config.encoder_type == "resnet" else CNN2DEncoder(in_ch, config.hidden_dim)
            self.frame_pool = nn.AdaptiveAvgPool2d(1)
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.gru_hidden,
            num_layers=config.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        self.out_dim = config.gru_hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps = x.shape[0], x.shape[1]
        rest = x.shape[2:]
        x_flat = x.reshape(bsz * steps, *rest)
        feat = self.encoder(x_flat)
        feat = self.frame_pool(feat).flatten(1)
        feat = feat.reshape(bsz, steps, -1)
        gru_out, _ = self.gru(feat)
        return gru_out


class StaticBetaNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.features = StaticFeatureExtractor(config)
        self.head = EvidentialHead(self.features.out_dim, config.num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        alpha, beta = self.head(feat)
        if return_features:
            return alpha, beta, feat
        return alpha, beta


class SequentialBetaNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.features = SequentialFeatureExtractor(config)
        self.head = EvidentialHead(self.features.out_dim, config.num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        alpha, beta = self.head(feat)
        if return_features:
            return alpha, beta, feat
        return alpha, beta


class StaticEDLNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.features = StaticFeatureExtractor(config)
        self.head = DirichletHead(self.features.out_dim, config.num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        evidence, alpha = self.head(feat)
        if return_features:
            return evidence, alpha, feat
        return evidence, alpha


class SequentialEDLNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.features = SequentialFeatureExtractor(config)
        self.head = DirichletHead(self.features.out_dim, config.num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        evidence, alpha = self.head(feat)
        if return_features:
            return evidence, alpha, feat
        return evidence, alpha


class StaticClassifierNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.features = StaticFeatureExtractor(config)
        self.head = ClassifierHead(self.features.out_dim, config.num_classes, config.classifier_dropout)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits


class SequentialClassifierNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.features = SequentialFeatureExtractor(config)
        self.head = ClassifierHead(self.features.out_dim, config.num_classes, config.classifier_dropout)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        logits = self.head(feat)
        if return_features:
            return logits, feat
        return logits


class SWAGPosterior:
    """Diagonal SWAG posterior statistics for lightweight posterior sampling."""

    def __init__(self, max_snapshots: int = 20):
        self.max_snapshots = max(1, int(max_snapshots))
        self.num_snapshots = 0
        self._snapshots = deque(maxlen=self.max_snapshots)
        self.mean_state: Dict[str, torch.Tensor] = {}
        self.sq_mean_state: Dict[str, torch.Tensor] = {}

    def collect_model(self, model: nn.Module):
        state = {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
        self._snapshots.append(state)
        self.num_snapshots = len(self._snapshots)
        self._recompute_moments()

    def _recompute_moments(self):
        if not self._snapshots:
            self.mean_state = {}
            self.sq_mean_state = {}
            return
        keys = self._snapshots[0].keys()
        self.mean_state = {}
        self.sq_mean_state = {}
        for key in keys:
            stacked = torch.stack([snap[key] for snap in self._snapshots], dim=0)
            self.mean_state[key] = stacked.mean(dim=0)
            self.sq_mean_state[key] = (stacked ** 2).mean(dim=0)

    def sample_into(self, model: nn.Module, scale: float = 1.0):
        if self.num_snapshots == 0:
            return
        named_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, mean in self.mean_state.items():
                if name not in named_params:
                    continue
                variance = torch.clamp(self.sq_mean_state[name] - mean ** 2, min=1e-30)
                std = torch.sqrt(variance)
                noise = torch.randn_like(mean) * std * scale
                named_params[name].copy_((mean + noise).to(named_params[name].device))

    def state_dict(self) -> Dict[str, object]:
        return {
            "max_snapshots": self.max_snapshots,
            "num_snapshots": self.num_snapshots,
            "snapshots": list(self._snapshots),
        }

    def load_state_dict(self, state: Optional[Dict[str, object]]):
        if not state:
            return
        self.max_snapshots = int(state.get("max_snapshots", self.max_snapshots))
        self._snapshots = deque(state.get("snapshots", []), maxlen=self.max_snapshots)
        self.num_snapshots = len(self._snapshots)
        self._recompute_moments()


def enable_mc_dropout(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def build_model(config: Config) -> nn.Module:
    """Build the configured model."""
    if config.method == "beta":
        model = StaticBetaNet(config) if config.mode == "static" else SequentialBetaNet(config)
    elif config.method == "edl":
        model = StaticEDLNet(config) if config.mode == "static" else SequentialEDLNet(config)
    elif config.method in ("bnn", "swag", "maxsoftmax"):
        model = StaticClassifierNet(config) if config.mode == "static" else SequentialClassifierNet(config)
    else:
        raise ValueError(f"Unknown method '{config.method}'. Expected one of: beta, edl, bnn, swag, maxsoftmax")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[Model] {model.__class__.__name__} | method={config.method} | "
        f"Encoder={config.encoder_type} | Dim={config.input_dim} | params={n_params:,}"
    )
    return model
