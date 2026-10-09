"""Loss functions for beta evidence, EDL, BNN, SWAG, and MaxSoftmax."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_class_dims(x: torch.Tensor) -> torch.Tensor:
    if x.dim() <= 2:
        return x
    return x.reshape(-1, x.shape[-1])


def _labels_to_indices(y_true: torch.Tensor) -> torch.Tensor:
    y_flat = _flatten_class_dims(y_true)
    return y_flat.argmax(dim=-1)


class BetaEvidenceLoss(nn.Module):
    """Beta evidential loss used by the existing method."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(alpha, min=1.0 + 1e-4, max=1e6)
        beta = torch.clamp(beta, min=1.0 + 1e-4, max=1e6)
        sum_ab = alpha + beta
        sum_ab = torch.clamp(sum_ab, min=2.0 + 2e-4, max=2e6)
        digamma_sum = torch.digamma(sum_ab)
        digamma_a = torch.digamma(alpha)
        digamma_b = torch.digamma(beta)

        nll_loss = -(y_true * (digamma_a - digamma_sum) + (1.0 - y_true) * (digamma_b - digamma_sum))

        if self.config is not None and getattr(self.config, "use_evidential_reweighting", False):
            vacuity = 2.0 / sum_ab
            vague = vacuity.max(dim=-1, keepdim=True)[0]
            gamma = getattr(self.config, "reweighting_gamma", 2.0)
            discount = torch.pow(1.0 - vague, gamma)
            if hasattr(self, "current_epoch"):
                max_epochs = getattr(self.config, "epochs", 1)
                if max_epochs > 0:
                    anneal_weight = (self.current_epoch / max_epochs) ** 2
                    discount = 1.0 - anneal_weight * (1.0 - discount)
            loss = (nll_loss * discount.detach()).mean()
        else:
            loss = nll_loss.mean()

        if self.config is not None and getattr(self.config, "use_uq_reg", False):
            b_pred = (alpha - 1.0) / sum_ab
            d_pred = (beta - 1.0) / sum_ab
            u_pred = 2.0 / sum_ab
            l_reg = torch.abs(y_true - b_pred) * (1.0 - u_pred + d_pred)
            lam = getattr(self.config, "uq_reg_lambda", 0.1)
            loss = loss + lam * l_reg.mean()

        return loss


class EDLClassificationLoss(nn.Module):
    """Standard Dirichlet EDL classification loss with annealed KL."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.current_epoch = 1

    @staticmethod
    def _dirichlet_kl(alpha: torch.Tensor) -> torch.Tensor:
        num_classes = alpha.size(-1)
        ones = torch.ones((1, num_classes), dtype=alpha.dtype, device=alpha.device)
        sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        first = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
            + torch.lgamma(ones).sum(dim=-1, keepdim=True)
            - torch.lgamma(torch.sum(ones, dim=-1, keepdim=True))
        )
        second = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1, keepdim=True)
        return first + second

    def forward(self, evidence: torch.Tensor, alpha: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_flat = _flatten_class_dims(y_true)
        alpha_flat = torch.clamp(_flatten_class_dims(alpha), min=1.0 + 1e-4, max=1e6)
        evidence_flat = torch.clamp(_flatten_class_dims(evidence), min=1e-4, max=1e6)

        sum_alpha = torch.clamp(alpha_flat.sum(dim=-1, keepdim=True), min=alpha_flat.shape[-1] + 1e-4, max=1e6)
        probs = alpha_flat / sum_alpha

        mse = torch.sum((y_flat - probs) ** 2, dim=-1, keepdim=True)
        var = torch.sum(alpha_flat * (sum_alpha - alpha_flat) / (sum_alpha * sum_alpha * (sum_alpha + 1.0)), dim=-1, keepdim=True)

        alpha_tilde = evidence_flat * (1.0 - y_flat) + 1.0
        kl = self._dirichlet_kl(alpha_tilde)
        anneal_epochs = max(1, getattr(self.config, "edl_annealing_epochs", 10))
        anneal = min(1.0, float(self.current_epoch) / anneal_epochs)
        return (mse + var + anneal * kl).mean()


class ClassificationLoss(nn.Module):
    """Cross-entropy classification loss used by BNN, SWAG, and MaxSoftmax."""

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        logits_flat = _flatten_class_dims(logits)
        targets = _labels_to_indices(y_true)
        return F.cross_entropy(logits_flat, targets)


def build_loss(config) -> nn.Module:
    if config.method == "beta":
        return BetaEvidenceLoss(config=config)
    if config.method == "edl":
        return EDLClassificationLoss(config=config)
    if config.method in ("bnn", "swag", "maxsoftmax"):
        return ClassificationLoss()
    raise ValueError(f"Unknown method '{config.method}'")


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy loss for optional UGA alignment."""

    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-l2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target, w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None):
        if source.dim() > 2:
            source = source.reshape(-1, source.size(-1))
        if target.dim() > 2:
            target = target.reshape(-1, target.size(-1))

        n_s = int(source.size()[0])
        n_t = int(target.size()[0])

        if w_s is not None:
            w_s = w_s.reshape(-1) / (torch.sum(w_s) + 1e-8)
        else:
            w_s = torch.ones(n_s, device=source.device) / n_s

        if w_t is not None:
            w_t = w_t.reshape(-1) / (torch.sum(w_t) + 1e-8)
        else:
            w_t = torch.ones(n_t, device=target.device) / n_t

        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        xx = kernels[:n_s, :n_s]
        yy = kernels[n_s:, n_s:]
        xy = kernels[:n_s, n_s:]
        yx = kernels[n_s:, :n_s]

        loss_xx = torch.sum(w_s.unsqueeze(1) * xx * w_s.unsqueeze(0))
        loss_yy = torch.sum(w_t.unsqueeze(1) * yy * w_t.unsqueeze(0))
        loss_xy = torch.sum(w_s.unsqueeze(1) * xy * w_t.unsqueeze(0))
        loss_yx = torch.sum(w_t.unsqueeze(1) * yx * w_s.unsqueeze(0))
        return loss_xx + loss_yy - loss_xy - loss_yx
