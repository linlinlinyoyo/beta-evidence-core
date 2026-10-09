"""Unified training entry for Beta, EDL, BNN(MC Dropout), SWAG, and MaxSoftmax."""

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from datasets.base import build_dataloaders
from inference import (
    compute_belief_metrics,
    compute_beta_method_metrics,
    compute_edl_method_metrics,
    compute_maxsoftmax_method_metrics,
    compute_probabilistic_method_metrics,
    max_softmax_score,
    ood_score_Upn_max,
    predictive_entropy,
    search_beta_hyperparams,
)
from loss import MMDLoss, build_loss
from model import SWAGPosterior, build_model, enable_mc_dropout
from visualize import plot_inference_report, plot_training_curves


def _sanitize_metrics(d: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        if isinstance(v, float) and np.isnan(v):
            v = None
        out[k] = v
    return out


def _score_key(method_name: str) -> str:
    return "score_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in method_name).strip("_")


def _scope_path_by_method(base_path: str, method: str) -> str:
    normalized = os.path.normpath(base_path)
    if os.path.basename(normalized).lower() == method.lower():
        return base_path
    return os.path.join(base_path, method)


class Trainer:
    def __init__(self, config: Config, dataloaders=None, class_names=None):
        self.config = config
        self.method = config.method
        self.device = config.get_device()

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        print(config.summary())
        if dataloaders is not None:
            self.dataloaders = dataloaders
            self.class_names = class_names
        else:
            self.dataloaders, self.class_names = build_dataloaders(config)

        self.model = build_model(config).to(self.device)
        self.criterion = build_loss(config)

        self.uga_strategy = config.uga_strategy if self.method == "beta" else "none"
        self.mmd_criterion = MMDLoss() if self.uga_strategy != "none" else None
        self.swag_posterior = SWAGPosterior(config.swag_max_snapshots) if self.method == "swag" else None

        if self.method == "swag":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step,
            gamma=config.scheduler_gamma,
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

    def _compute_beta_loss(self, batch_x: torch.Tensor, batch_y: torch.Tensor, epoch: int, target_iter=None):
        self.criterion.current_epoch = epoch
        use_uga = self.uga_strategy != "none" and "target_train" in self.dataloaders

        if use_uga and self.uga_strategy == "feature":
            alpha, beta, feat_s = self.model(batch_x, return_features=True)
        else:
            alpha, beta = self.model(batch_x)

        loss = self.criterion(alpha, beta, batch_y)
        if getattr(self.config, "use_pseudo_ood_reg", False):
            pseudo_x = self._make_pseudo_ood(batch_x)
            pseudo_alpha, pseudo_beta = self.model(pseudo_x)
            pseudo_sum = torch.clamp(pseudo_alpha + pseudo_beta, min=2.0 + 2e-4, max=2e6)
            pseudo_penalty = torch.relu(pseudo_sum - 2.0).mean()
            loss = loss + getattr(self.config, "pseudo_ood_reg_lambda", 0.01) * pseudo_penalty

        if not use_uga:
            return loss, target_iter

        try:
            batch_target_x, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(self.dataloaders["target_train"])
            batch_target_x, _ = next(target_iter)

        batch_target_x = batch_target_x.to(self.device)

        if self.uga_strategy == "feature":
            alpha_t, beta_t, feat_t = self.model(batch_target_x, return_features=True)
            source_uga = torch.cat([feat_s, alpha, beta], dim=-1)
            target_uga = torch.cat([feat_t, alpha_t, beta_t], dim=-1)
        else:
            alpha_t, beta_t = self.model(batch_target_x)
            source_uga = torch.cat([alpha, beta], dim=-1)
            target_uga = torch.cat([alpha_t, beta_t], dim=-1)

        source_uga = source_uga.reshape(-1, source_uga.size(-1))
        target_uga = target_uga.reshape(-1, target_uga.size(-1))
        u_t = (2.0 / (alpha_t + beta_t)).mean(dim=-1).reshape(-1)
        w_t = torch.clamp(1.0 - u_t, min=0.0)
        loss_mmd = self.mmd_criterion(source_uga, target_uga, w_t=w_t)
        return loss + getattr(self.config, "uga_lambda", 1.0) * loss_mmd, target_iter

    def _make_pseudo_ood(self, batch_x: torch.Tensor) -> torch.Tensor:
        if batch_x.size(0) < 2:
            noise_std = getattr(self.config, "pseudo_ood_noise_std", 0.05)
            return batch_x + noise_std * torch.randn_like(batch_x)
        perm = torch.randperm(batch_x.size(0), device=batch_x.device)
        lam = torch.rand(batch_x.size(0), device=batch_x.device)
        view_shape = [batch_x.size(0)] + [1] * (batch_x.dim() - 1)
        lam = lam.view(*view_shape)
        mixed = lam * batch_x + (1.0 - lam) * batch_x[perm]
        noise_std = getattr(self.config, "pseudo_ood_noise_std", 0.05)
        return mixed + noise_std * torch.randn_like(mixed)

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        target_iter = iter(self.dataloaders["target_train"]) if self.uga_strategy != "none" and "target_train" in self.dataloaders else None

        for batch_x, batch_y in self.dataloaders["train"]:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            self.optimizer.zero_grad()

            if self.method == "beta":
                loss, target_iter = self._compute_beta_loss(batch_x, batch_y, epoch, target_iter)
            elif self.method == "edl":
                self.criterion.current_epoch = epoch
                evidence, alpha = self.model(batch_x)
                loss = self.criterion(evidence, alpha, batch_y)
            else:
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_x, batch_y in self.dataloaders["val"]:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            if self.method == "beta":
                self.criterion.current_epoch = epoch
                alpha, beta = self.model(batch_x)
                loss = self.criterion(alpha, beta, batch_y)
            elif self.method == "edl":
                self.criterion.current_epoch = epoch
                evidence, alpha = self.model(batch_x)
                loss = self.criterion(evidence, alpha, batch_y)
            else:
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _maybe_collect_swag(self, epoch: int):
        if self.method != "swag" or self.swag_posterior is None:
            return
        if epoch < self.config.swag_start_epoch:
            return
        if (epoch - self.config.swag_start_epoch) % max(1, self.config.swag_collect_freq) != 0:
            return
        self.swag_posterior.collect_model(self.model)
        print(f"  [SWAG] collected snapshot #{self.swag_posterior.num_snapshots}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
        }
        if self.swag_posterior is not None:
            state["swag_state"] = self.swag_posterior.state_dict()

        latest_path = os.path.join(self.config.checkpoint_dir, "latest.pth")
        torch.save(state, latest_path)

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"  [Checkpoint] saved best model at epoch {epoch}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if self.swag_posterior is not None:
            self.swag_posterior.load_state_dict(checkpoint.get("swag_state"))
        print(f"[Train] loaded checkpoint: {path}")

    def train(self):
        print("\n" + "=" * 60)
        print("Start training")
        print("=" * 60)
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_one_epoch(epoch)
            self.train_losses.append(train_loss)

            val_loss = self.validate(epoch)
            self.val_losses.append(val_loss)

            self.scheduler.step()
            self._maybe_collect_swag(epoch)

            current_lr = self.scheduler.get_last_lr()[0]
            epoch_time = time.time() - epoch_start
            print(
                f"  Epoch {epoch:3d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
            )

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best=is_best)

            if self.patience_counter >= self.config.patience:
                print(f"\n  [Early Stop] no improvement for {self.config.patience} epochs.")
                print(f"  Best epoch: {self.best_epoch}, best val loss: {self.best_val_loss:.6f}")
                break

        total_time = time.time() - start_time
        print(f"\nTraining finished in {total_time:.1f}s")
        print(f"Best model: epoch {self.best_epoch}, val loss {self.best_val_loss:.6f}")

        plot_training_curves(self.train_losses, self.val_losses, self.config.results_dir)
        self._save_log()
        self.evaluate()

    def _save_log(self):
        log = {
            "config": self.config.__dict__,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }
        path = os.path.join(self.config.log_dir, "training_log.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f"[Train] saved training log: {path}")

    def _preferred_checkpoint_path(self) -> str:
        if self.method == "swag":
            return os.path.join(self.config.checkpoint_dir, "latest.pth")
        return os.path.join(self.config.checkpoint_dir, "best_model.pth")

    def _flatten_eval_tensors(self, *arrays):
        flattened = []
        for arr in arrays:
            if arr.ndim == 3:
                flattened.append(arr.reshape(-1, arr.shape[-1]))
            else:
                flattened.append(arr)
        return flattened

    def _predict_mc_probs(self, batch_x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        enable_mc_dropout(self.model)
        probs_list = []
        with torch.no_grad():
            for _ in range(self.config.mc_dropout_samples):
                logits = self.model(batch_x)
                probs_list.append(torch.softmax(logits, dim=-1))
        self.model.eval()
        mean_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        return mean_probs.cpu().numpy()

    def _snapshot_trainable_state(self):
        return {name: param.detach().clone() for name, param in self.model.named_parameters() if param.requires_grad}

    def _restore_trainable_state(self, state: Dict[str, torch.Tensor]):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state:
                    param.copy_(state[name])

    def _predict_swag_probs(self, batch_x: torch.Tensor) -> np.ndarray:
        if self.swag_posterior is None or self.swag_posterior.num_snapshots == 0:
            logits = self.model(batch_x)
            return torch.softmax(logits, dim=-1).cpu().numpy()

        original_state = self._snapshot_trainable_state()
        probs_list = []
        with torch.no_grad():
            for _ in range(self.config.swag_samples):
                self.swag_posterior.sample_into(self.model, scale=self.config.swag_scale)
                self.model.eval()
                logits = self.model(batch_x)
                probs_list.append(torch.softmax(logits, dim=-1))
        self._restore_trainable_state(original_state)
        self.model.eval()
        mean_probs = torch.stack(probs_list, dim=0).mean(dim=0)
        return mean_probs.cpu().numpy()

    @torch.no_grad()
    def evaluate(self):
        ckpt_path = self._preferred_checkpoint_path()
        if os.path.exists(ckpt_path):
            self.load_checkpoint(ckpt_path)

        self.model.eval()
        eval_loaders = [("source_test", "test")]
        if "target_test" in self.dataloaders:
            eval_loaders.append(("target_test", "target_test"))

        for display_name, loader_name in eval_loaders:
            print("\n" + "=" * 60)
            print(f"Evaluate: {display_name}")
            print("=" * 60)

            test_loss = 0.0
            n_batches = 0
            all_labels = []
            beta_alphas = []
            beta_betas = []
            edl_alphas = []
            prob_outputs = []
            feature_outputs = []
            vacuity_timeline = []
            belief_timeline = []
            disbelief_timeline = []

            for batch_x, batch_y in self.dataloaders[loader_name]:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                all_labels.append(batch_y.cpu().numpy())

                if self.method == "beta":
                    alpha, beta = self.model(batch_x)
                    loss = self.criterion(alpha, beta, batch_y)
                    metrics = compute_belief_metrics(alpha, beta)
                    beta_alphas.append(alpha.cpu().numpy())
                    beta_betas.append(beta.cpu().numpy())
                    vacuity_timeline.append(metrics["vacuity"].cpu().numpy())
                    belief_timeline.append(metrics["belief"].cpu().numpy())
                    disbelief_timeline.append(metrics["disbelief"].cpu().numpy())
                elif self.method == "edl":
                    evidence, alpha = self.model(batch_x)
                    loss = self.criterion(evidence, alpha, batch_y)
                    edl_alphas.append(alpha.cpu().numpy())
                elif self.method == "bnn":
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)
                    prob_outputs.append(self._predict_mc_probs(batch_x))
                elif self.method == "maxsoftmax":
                    logits, features = self.model(batch_x, return_features=True)
                    loss = self.criterion(logits, batch_y)
                    prob_outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                    feature_outputs.append(features.cpu().numpy())
                else:
                    logits = self.model(batch_x)
                    loss = self.criterion(logits, batch_y)
                    prob_outputs.append(self._predict_swag_probs(batch_x))

                test_loss += loss.item()
                n_batches += 1

            test_loss /= max(n_batches, 1)
            print(f"  Test loss: {test_loss:.6f}")

            labels = np.concatenate(all_labels, axis=0)
            labels = self._flatten_eval_tensors(labels)[0]
            is_ood_for_artifacts = labels.sum(axis=-1) <= 0.5

            if self.method == "beta":
                alphas = np.concatenate(beta_alphas, axis=0)
                betas = np.concatenate(beta_betas, axis=0)
                alphas, betas = self._flatten_eval_tensors(alphas, betas)

                hp_results = search_beta_hyperparams(
                    alphas,
                    betas,
                    labels,
                    objective=getattr(self.config, "beta_search_objective", "auroc"),
                )
                strategy_params = {name: info["best_params"] for name, info in hp_results.items()}
                all_metrics = compute_beta_method_metrics(alphas, betas, labels, strategy_params=strategy_params)

                vacuity = np.concatenate(vacuity_timeline, axis=0)
                belief = np.concatenate(belief_timeline, axis=0)
                disbelief = np.concatenate(disbelief_timeline, axis=0)
                vacuity, belief, disbelief = self._flatten_eval_tensors(vacuity, belief, disbelief)

                label_sum = labels.sum(axis=-1)
                is_ood = ~(label_sum > 0.5)
                ood_boundary_idx = int(np.argmax(is_ood)) if is_ood.sum() > 0 and (~is_ood).sum() > 0 else None

                results = {
                    "vacuity_timeline": vacuity,
                    "belief_timeline": belief,
                    "disbelief_timeline": disbelief,
                    "warnings": [],
                    "ood_boundary_idx": ood_boundary_idx,
                }
                suffix_dir = os.path.join(self.config.results_dir, loader_name)
                os.makedirs(suffix_dir, exist_ok=True)
                plot_inference_report(results, suffix_dir, self.config, class_names=self.class_names)
                probs_for_artifacts = alphas / (alphas + betas)
                preds_for_artifacts = alphas.argmax(axis=-1)
                artifact_payload = {
                    "labels": labels,
                    "is_ood": is_ood_for_artifacts,
                    "probs": probs_for_artifacts,
                    "preds": preds_for_artifacts,
                    "alpha": alphas,
                    "beta": betas,
                }
                artifact_payload[_score_key("Beta+MaxVac")] = (2.0 / (alphas + betas)).max(axis=-1)
                upn_params = strategy_params.get("Beta+Upn_max", {})
                artifact_payload[_score_key("Beta+Upn_max")] = ood_score_Upn_max(alphas, betas, **upn_params)
            elif self.method == "edl":
                alpha = np.concatenate(edl_alphas, axis=0)
                alpha = self._flatten_eval_tensors(alpha)[0]
                all_metrics = compute_edl_method_metrics(alpha, labels)
                hp_results = {}
                probs_for_artifacts = alpha / np.sum(alpha, axis=-1, keepdims=True)
                preds_for_artifacts = probs_for_artifacts.argmax(axis=-1)
                artifact_payload = {
                    "labels": labels,
                    "is_ood": is_ood_for_artifacts,
                    "probs": probs_for_artifacts,
                    "preds": preds_for_artifacts,
                    "alpha": alpha,
                    _score_key("EDL"): alpha.shape[-1] / np.sum(alpha, axis=-1),
                }
            elif self.method == "bnn":
                probs = np.concatenate(prob_outputs, axis=0)
                probs = self._flatten_eval_tensors(probs)[0]
                all_metrics = compute_probabilistic_method_metrics("BNN", probs, labels)
                hp_results = {}
                artifact_payload = {
                    "labels": labels,
                    "is_ood": is_ood_for_artifacts,
                    "probs": probs,
                    "preds": probs.argmax(axis=-1),
                    _score_key("BNN"): predictive_entropy(probs),
                }
            elif self.method == "maxsoftmax":
                probs = np.concatenate(prob_outputs, axis=0)
                probs = self._flatten_eval_tensors(probs)[0]
                all_metrics = compute_maxsoftmax_method_metrics(probs, labels)
                hp_results = {}
                artifact_payload = {
                    "labels": labels,
                    "is_ood": is_ood_for_artifacts,
                    "probs": probs,
                    "preds": probs.argmax(axis=-1),
                    _score_key("MaxSoftmax"): max_softmax_score(probs),
                }
                if feature_outputs:
                    features = np.concatenate(feature_outputs, axis=0)
                    artifact_payload["features"] = self._flatten_eval_tensors(features)[0]
            else:
                probs = np.concatenate(prob_outputs, axis=0)
                probs = self._flatten_eval_tensors(probs)[0]
                all_metrics = compute_probabilistic_method_metrics("SWAG", probs, labels)
                hp_results = {}
                artifact_payload = {
                    "labels": labels,
                    "is_ood": is_ood_for_artifacts,
                    "probs": probs,
                    "preds": probs.argmax(axis=-1),
                    _score_key("SWAG"): predictive_entropy(probs),
                }

            if all_metrics:
                print("\n  ============== Metrics ==============")
                for name, metrics in all_metrics.items():
                    print(
                        f"  {name:<14} | "
                        f"Acc={metrics.get('Acc', float('nan'))*100:.2f}% | "
                        f"AUROC={metrics.get('AUROC', float('nan'))*100:.2f}% | "
                        f"AUPR={metrics.get('AUPR', float('nan'))*100:.2f}% | "
                        f"FPR95={metrics.get('FPR@95TPR', float('nan'))*100:.2f}% | "
                        f"AURC={metrics.get('AURC', float('nan')):.4f}"
                    )

            suffix_dir = os.path.join(self.config.results_dir, loader_name)
            os.makedirs(suffix_dir, exist_ok=True)
            metrics_save = {name: _sanitize_metrics(metrics) for name, metrics in all_metrics.items()}
            if hp_results:
                metrics_save["_hyperparams_search"] = hp_results
            metrics_path = os.path.join(suffix_dir, "evaluation_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_save, f, indent=2, ensure_ascii=False)
            print(f"  [Evaluate] saved metrics: {metrics_path}")
            artifacts_path = os.path.join(suffix_dir, "analysis_artifacts.npz")
            np.savez_compressed(artifacts_path, **artifact_payload)
            print(f"  [Evaluate] saved analysis artifacts: {artifacts_path}")


def _set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _save_summary_csv(path: str, rows: list, metric_keys: list):
    import csv

    fieldnames = ["group", "method"]
    for mk in metric_keys:
        fieldnames.extend([mk + "_mean", mk + "_std"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_multi_seed(base_config: Config, n_repeats: int, start_seed: int, eval_only: bool = False):
    seeds = [start_seed + run_idx for run_idx in range(n_repeats)]
    run_seed_list(base_config, seeds, eval_only=eval_only)


def run_seed_list(base_config: Config, seeds: list, eval_only: bool = False):
    all_run_metrics = []
    base_ckpt = _scope_path_by_method(base_config.checkpoint_dir, base_config.method)
    base_results = _scope_path_by_method(base_config.results_dir, base_config.method)
    base_log = _scope_path_by_method(base_config.log_dir, base_config.method)
    shared_dataloaders = None
    shared_class_names = None

    if base_config.shared_split_seed is not None:
        print(f"\n  [Multi-seed] use fixed seed {base_config.shared_split_seed} for the shared data split...")
        _set_seed(base_config.shared_split_seed)
        shared_dataloaders, shared_class_names = build_dataloaders(base_config)
    else:
        print("\n  [Multi-seed] rebuild data split for each seed (legacy-compatible mode)...")

    for run_idx, seed in enumerate(seeds):
        print("\n" + "#" * 70)
        print(f"#  Repeat {run_idx + 1}/{len(seeds)} | seed = {seed}")
        print("#" * 70)

        _set_seed(seed)
        config = Config(**{k: v for k, v in base_config.__dict__.items()})
        config.checkpoint_dir = os.path.join(base_ckpt, f"seed_{seed}")
        config.results_dir = os.path.join(base_results, f"seed_{seed}")
        config.log_dir = os.path.join(base_log, f"seed_{seed}")

        os.makedirs(config.log_dir, exist_ok=True)
        config.save(os.path.join(config.log_dir, "config.json"))

        if shared_dataloaders is None:
            trainer = Trainer(config)
        else:
            trainer = Trainer(config, dataloaders=shared_dataloaders, class_names=shared_class_names)
        if eval_only:
            trainer.evaluate()
        else:
            trainer.train()

        for loader_name in ["test", "target_test"]:
            metrics_path = os.path.join(config.results_dir, loader_name, "evaluation_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    run_result = json.load(f)
                run_result = {k: v for k, v in run_result.items() if not k.startswith("_")}
                all_run_metrics.append({"seed": seed, "loader": loader_name, "data": run_result})

    if not all_run_metrics:
        print("\n[Warning] no evaluation metrics were collected.")
        return

    print("\n" + "=" * 90)
    print("  Multi-seed summary (mean ± std)")
    print("=" * 90)

    loaders_seen = sorted(set(r["loader"] for r in all_run_metrics))
    metric_keys = ["Acc", "AUROC", "AUPR", "FPR@95TPR", "AURC", "Risk@80Cov", "Cov@5Risk"]
    group_map = {
        "Beta+Upn_max": "Final",
        "Beta+MaxVac": "Internal",
        "EDL": "External",
        "BNN": "External",
        "SWAG": "External",
        "MaxSoftmax": "External",
    }

    for loader_name in loaders_seen:
        loader_runs = [r["data"] for r in all_run_metrics if r["loader"] == loader_name]
        if not loader_runs:
            continue

        print(f"\n  Split: {loader_name} ({len(loader_runs)} runs)")
        print("  " + "-" * 85)

        method_names = list(loader_runs[0].keys())
        rows = []
        for method_name in method_names:
            row = {"group": group_map.get(method_name, "Other"), "method": method_name}
            for mk in metric_keys:
                vals = [run[method_name][mk] for run in loader_runs if method_name in run and run[method_name].get(mk) is not None]
                if vals:
                    row[mk + "_mean"] = float(np.mean(vals))
                    row[mk + "_std"] = float(np.std(vals) if len(vals) > 1 else 0.0)
                else:
                    row[mk + "_mean"] = None
                    row[mk + "_std"] = None
            rows.append(row)

        for row in rows:
            auroc = row["AUROC_mean"] * 100 if row["AUROC_mean"] is not None else float("nan")
            aurc = row["AURC_mean"] if row["AURC_mean"] is not None else float("nan")
            print(f"  {row['group']:<8} {row['method']:<14} AUROC={auroc:.2f}% | AURC={aurc:.4f}")

        summary_dir = os.path.join(base_results, "summary")
        os.makedirs(summary_dir, exist_ok=True)
        csv_path = os.path.join(summary_dir, f"{loader_name}_summary.csv")
        _save_summary_csv(csv_path, rows, metric_keys)
        print(f"  [Summary] saved {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Train uncertainty methods for the beta evidence benchmark.")
    parser.add_argument("--mode", type=str, default=None, choices=["sequential", "static"])
    parser.add_argument("--method", type=str, default=None, choices=["beta", "edl", "bnn", "swag", "maxsoftmax"])

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--file_format", type=str, default=None, choices=["csv", "mat", "npy"])
    parser.add_argument("--csv_column", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--sample_length", type=int, default=None)
    parser.add_argument("--input_dim", type=str, default=None, choices=["1d", "2d"])
    parser.add_argument("--split_strategy", type=str, default=None, choices=["random", "stratified_group", "stratified_window"])

    parser.add_argument("--encoder_type", type=str, default=None, choices=["cnn", "resnet", "multiscale_resnet1d"])
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--classifier_dropout", type=float, default=None)
    parser.add_argument("--mc_dropout_samples", type=int, default=None)
    parser.add_argument("--swag_samples", type=int, default=None)
    parser.add_argument("--swag_start_epoch", type=int, default=None)
    parser.add_argument("--swag_collect_freq", type=int, default=None)
    parser.add_argument("--swag_max_snapshots", type=int, default=None)
    parser.add_argument("--edl_annealing_epochs", type=int, default=None)
    parser.add_argument("--disable_evidential_reweighting", action="store_true")
    parser.add_argument("--reweighting_gamma", type=float, default=None)
    parser.add_argument("--use_uq_reg", action="store_true", default=None)
    parser.add_argument("--uq_reg_lambda", type=float, default=None)
    parser.add_argument("--use_pseudo_ood_reg", action="store_true", default=None)
    parser.add_argument("--pseudo_ood_reg_lambda", type=float, default=None)
    parser.add_argument("--pseudo_ood_noise_std", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--v_early", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--shared_split_seed", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)

    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_list", nargs="+", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    if args.config_file and os.path.exists(args.config_file):
        config = Config.load(args.config_file)
        print(f"[Train] loaded config from {args.config_file}")
    else:
        config = Config()

    for key, value in vars(args).items():
        if value is not None and key not in ("config_file", "resume", "repeat", "seed", "disable_evidential_reweighting") and hasattr(config, key):
            setattr(config, key, value)
    if args.disable_evidential_reweighting:
        config.use_evidential_reweighting = False

    if args.seed_list or args.repeat > 1:
        config.checkpoint_dir = _scope_path_by_method(config.checkpoint_dir, config.method)
        config.results_dir = _scope_path_by_method(config.results_dir, config.method)
        config.log_dir = _scope_path_by_method(config.log_dir, config.method)
        if args.seed_list:
            run_seed_list(config, args.seed_list, eval_only=args.eval_only)
        else:
            run_multi_seed(config, args.repeat, args.seed, eval_only=args.eval_only)
    else:
        config.checkpoint_dir = _scope_path_by_method(config.checkpoint_dir, config.method)
        config.results_dir = _scope_path_by_method(config.results_dir, config.method)
        config.log_dir = _scope_path_by_method(config.log_dir, config.method)
        config_save_path = os.path.join(config.log_dir, "config.json")
        os.makedirs(config.log_dir, exist_ok=True)
        config.save(config_save_path)
        print(f"[Train] saved config: {config_save_path}")

        split_seed = config.shared_split_seed if config.shared_split_seed is not None else args.seed
        _set_seed(split_seed)
        print(f"[Train] seeded single-run dataloaders/model with seed={split_seed}")

        trainer = Trainer(config)
        if args.resume:
            trainer.load_checkpoint(args.resume)
        if args.eval_only:
            trainer.evaluate()
        else:
            trainer.train()


if __name__ == "__main__":
    main()
