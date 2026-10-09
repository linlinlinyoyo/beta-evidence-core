"""Inference utilities and unified evaluation metrics."""

from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch

from config import Config


def compute_belief_metrics(alpha: torch.Tensor, beta: torch.Tensor) -> Dict[str, torch.Tensor]:
    sum_ab = alpha + beta
    return {
        "belief": (alpha - 1.0) / sum_ab,
        "disbelief": (beta - 1.0) / sum_ab,
        "vacuity": 2.0 / sum_ab,
        "probability": alpha / sum_ab,
    }


def ood_score_max_vacuity(alphas: np.ndarray, betas: np.ndarray, **kwargs) -> np.ndarray:
    vacuity = 2.0 / (alphas + betas)
    return vacuity.max(axis=-1)


def ood_score_Upn_max(alphas: np.ndarray, betas: np.ndarray, lam1: float = 0.5, **kwargs) -> np.ndarray:
    up = 1.0 / (np.max(alphas, axis=-1) + 1e-8)
    un = 1.0 - np.max(1.0 / betas, axis=-1)
    return lam1 * up + (1.0 - lam1) * un


def _compute_aurc_for_scores(ood_scores: np.ndarray, preds: np.ndarray, labels: np.ndarray, is_id: np.ndarray) -> float:
    if is_id.sum() < 2:
        return 1.0
    id_scores = ood_scores[is_id]
    id_labels_idx = labels[is_id].argmax(axis=-1)
    id_correct = preds[is_id] == id_labels_idx

    sorted_idx = np.argsort(id_scores)
    sorted_correct = id_correct[sorted_idx]
    n_id = int(is_id.sum())
    cum_correct = np.cumsum(sorted_correct)
    k_arr = np.arange(1, n_id + 1)
    risks = 1.0 - cum_correct / k_arr
    coverages = k_arr / n_id
    return float(np.trapz(risks, coverages))


def search_beta_hyperparams(
    alphas: np.ndarray,
    betas: np.ndarray,
    labels: np.ndarray,
    objective: str = "auroc",
) -> Dict[str, Dict]:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("[Warning] scikit-learn is required for hyper-parameter search.")
        return {}

    label_sum = labels.sum(axis=-1)
    is_id = label_sum > 0.5
    is_ood = ~is_id
    if is_id.sum() == 0 or is_ood.sum() == 0:
        return {}

    preds = alphas.argmax(axis=-1)
    ood_targets = is_ood.astype(int)
    grids = np.arange(0, 1.05, 0.1)
    best_obj = -np.inf
    best = {"best_params": {"lam1": 0.5}, "best_auroc": 0.0, "best_aurc": 1.0, "best_objective": -np.inf}

    for lam1 in grids:
        if lam1 == 0.0:
            continue
        scores = ood_score_Upn_max(alphas, betas, lam1=lam1)
        try:
            auroc = roc_auc_score(ood_targets, scores)
        except ValueError:
            continue
        aurc = _compute_aurc_for_scores(scores, preds, labels, is_id)
        if objective == "auroc_minus_aurc":
            obj = auroc - aurc
        else:
            obj = auroc
        if obj > best_obj:
            best_obj = obj
            best = {
                "best_params": {"lam1": round(float(lam1), 4)},
                "best_auroc": float(auroc),
                "best_aurc": float(aurc),
                "best_objective": float(obj),
            }
    return {"Beta+Upn_max": best}


def compute_metrics_for_method(
    probs: np.ndarray,
    preds: np.ndarray,
    ood_scores: np.ndarray,
    labels: np.ndarray,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    except ImportError:
        print("[Warning] scikit-learn is required for metric computation.")
        return {}

    eps = 1e-12
    label_sum = labels.sum(axis=-1)
    is_id = label_sum > 0.5
    is_ood = ~is_id
    has_ood = is_id.sum() > 0 and is_ood.sum() > 0

    result: Dict[str, object] = {"_params": params if params else "default"}

    if is_id.sum() > 0:
        id_labels = labels[is_id]
        id_probs = probs[is_id]
        true_classes = id_labels.argmax(axis=-1)
        pred_classes = preds[is_id]
        result["Acc"] = float((true_classes == pred_classes).mean())

        nll = 0.0
        for i, c in enumerate(true_classes):
            nll -= np.log(id_probs[i, c] + eps)
        result["NLL"] = float(nll / max(len(true_classes), 1))
        result["Brier"] = float(np.mean(np.sum((id_probs - id_labels) ** 2, axis=-1)))

        confidences = np.max(id_probs, axis=-1)
        correctness = (pred_classes == true_classes).astype(float)
        num_bins = 15
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        n_id = int(is_id.sum())
        for m in range(num_bins):
            lo, hi = bin_boundaries[m], bin_boundaries[m + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / n_id) * abs(correctness[mask].mean() - confidences[mask].mean())
        result["ECE"] = float(ece)

        sorted_idx = np.argsort(confidences)
        bin_size = max(1, n_id // num_bins)
        aece = 0.0
        for m in range(num_bins):
            lo_i = m * bin_size
            hi_i = min((m + 1) * bin_size, n_id)
            if lo_i >= n_id:
                break
            idx = sorted_idx[lo_i:hi_i]
            aece += (len(idx) / n_id) * abs(correctness[idx].mean() - confidences[idx].mean())
        result["aECE"] = float(aece)
    else:
        for key in ("Acc", "NLL", "Brier", "ECE", "aECE"):
            result[key] = float("nan")

    if has_ood:
        ood_targets = is_ood.astype(int)
        result["AUROC"] = float(roc_auc_score(ood_targets, ood_scores))
        result["AUPR"] = float(average_precision_score(ood_targets, ood_scores))
        fpr_arr, tpr_arr, _ = roc_curve(ood_targets, ood_scores)
        idx95 = np.where(tpr_arr >= 0.95)[0]
        result["FPR@95TPR"] = float(fpr_arr[idx95[0]] if len(idx95) > 0 else 1.0)
    else:
        for key in ("AUROC", "AUPR", "FPR@95TPR"):
            result[key] = float("nan")

    if is_id.sum() > 0:
        id_scores = ood_scores[is_id]
        id_labels_idx = labels[is_id].argmax(axis=-1)
        id_correct = preds[is_id] == id_labels_idx
        sorted_idx = np.argsort(id_scores)
        sorted_correct = id_correct[sorted_idx]
        n_id = int(is_id.sum())

        coverages = []
        risks = []
        for k in range(1, n_id + 1):
            cov = k / n_id
            risk = 1.0 - sorted_correct[:k].mean()
            coverages.append(cov)
            risks.append(risk)
        coverages = np.array(coverages)
        risks = np.array(risks)
        result["AURC"] = float(np.trapz(risks, coverages))
        idx_80 = np.argmin(np.abs(coverages - 0.8))
        result["Risk@80Cov"] = float(risks[idx_80])
        safe_idx = np.where(risks <= 0.05)[0]
        result["Cov@5Risk"] = float(coverages[safe_idx[-1]] if len(safe_idx) > 0 else 0.0)
        result["_curve"] = {
            "coverage": coverages.tolist(),
            "risk": risks.tolist(),
        }
    else:
        for key in ("AURC", "Risk@80Cov", "Cov@5Risk"):
            result[key] = float("nan")

    return result


def compute_beta_method_metrics(
    alphas: np.ndarray,
    betas: np.ndarray,
    labels: np.ndarray,
    strategy_params: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict]:
    if strategy_params is None:
        strategy_params = {}

    probs = alphas / (alphas + betas)
    preds = alphas.argmax(axis=-1)

    maxvac_scores = ood_score_max_vacuity(alphas, betas)
    upn_params = strategy_params.get("Beta+Upn_max", {})
    upn_scores = ood_score_Upn_max(alphas, betas, **upn_params)

    return {
        "Beta+MaxVac": compute_metrics_for_method(probs, preds, maxvac_scores, labels, params={}),
        "Beta+Upn_max": compute_metrics_for_method(probs, preds, upn_scores, labels, params=upn_params),
    }


def compute_edl_method_metrics(alpha: np.ndarray, labels: np.ndarray) -> Dict[str, Dict]:
    probs = alpha / np.sum(alpha, axis=-1, keepdims=True)
    preds = probs.argmax(axis=-1)
    num_classes = alpha.shape[-1]
    vacuity = num_classes / np.sum(alpha, axis=-1)
    return {"EDL": compute_metrics_for_method(probs, preds, vacuity, labels)}


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def max_softmax_score(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    return 1.0 - np.max(probs, axis=-1)


def compute_probabilistic_method_metrics(name: str, probs: np.ndarray, labels: np.ndarray) -> Dict[str, Dict]:
    preds = probs.argmax(axis=-1)
    entropy = predictive_entropy(probs)
    return {name: compute_metrics_for_method(probs, preds, entropy, labels)}


def compute_maxsoftmax_method_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Dict]:
    preds = probs.argmax(axis=-1)
    return {"MaxSoftmax": compute_metrics_for_method(probs, preds, max_softmax_score(probs), labels)}


def check_early_warning(vacuity_history: List[np.ndarray], v_early: float, consecutive_windows: int) -> Dict[str, object]:
    if len(vacuity_history) < consecutive_windows:
        return {
            "warning": False,
            "warning_classes": [],
            "max_vacuity": vacuity_history[-1] if vacuity_history else None,
        }

    recent = vacuity_history[-consecutive_windows:]
    num_classes = recent[0].shape[0]
    warning_classes = []
    for k in range(num_classes):
        if all(v[k] > v_early for v in recent):
            warning_classes.append(k)

    return {
        "warning": len(warning_classes) > 0,
        "warning_classes": warning_classes,
        "max_vacuity": recent[-1],
    }


class EvidentialInferencer:
    def __init__(self, model: torch.nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = config.get_device()
        self.model.to(self.device)
        self.model.eval()
        self.vacuity_history: List[np.ndarray] = []
        self.belief_history: List[np.ndarray] = []
        self.disbelief_history: List[np.ndarray] = []

    def reset(self):
        self.vacuity_history.clear()
        self.belief_history.clear()
        self.disbelief_history.clear()

    @torch.no_grad()
    def infer_sequence(self, x: torch.Tensor) -> Dict[str, object]:
        x = x.to(self.device)
        alpha, beta = self.model(x)
        metrics = compute_belief_metrics(alpha, beta)
        belief = metrics["belief"][0].cpu().numpy()
        disbelief = metrics["disbelief"][0].cpu().numpy()
        vacuity = metrics["vacuity"][0].cpu().numpy()
        probability = metrics["probability"][0].cpu().numpy()

        warnings = []
        for t in range(vacuity.shape[0]):
            self.vacuity_history.append(vacuity[t])
            self.belief_history.append(belief[t])
            self.disbelief_history.append(disbelief[t])
            result = check_early_warning(self.vacuity_history, self.config.v_early, self.config.consecutive_windows)
            if result["warning"]:
                warnings.append({"frame": t, "classes": result["warning_classes"], "vacuity": vacuity[t].tolist()})

        return {
            "belief": belief,
            "disbelief": disbelief,
            "vacuity": vacuity,
            "probability": probability,
            "alpha": alpha[0].cpu().numpy(),
            "beta": beta[0].cpu().numpy(),
            "warnings": warnings,
        }


class OOD_EarlyWarning_Inferencer:
    def __init__(self, model: torch.nn.Module, config: Config):
        self.model = model
        self.config = config
        self.device = config.get_device()
        self.model.to(self.device)
        self.model.eval()
        self.vacuity_history: List[np.ndarray] = []
        self.belief_history: List[np.ndarray] = []
        self.disbelief_history: List[np.ndarray] = []
        self.step_count = 0

    def reset(self):
        self.vacuity_history.clear()
        self.belief_history.clear()
        self.disbelief_history.clear()
        self.step_count = 0


def build_inferencer(model: torch.nn.Module, config: Config):
    return EvidentialInferencer(model, config) if config.mode == "sequential" else OOD_EarlyWarning_Inferencer(model, config)
