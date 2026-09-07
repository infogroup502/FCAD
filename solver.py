from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_factory.data_loader import get_loader_segment
from model.FCAD import GaussFuzzyAnomalyModel, make_random_mask, masked_mse_loss


def adjust_learning_rate(optimizer, epoch, lr_):
    lr = lr_ * (0.5 ** ((epoch - 1) // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class Solver(object):
    DEFAULTS = {
        "stage23_lr": 1e-3,
        "stage2_num_patterns": 4,
        "stage2_center_init_std": 0.5,
        "stage2_sigma_init": 1.0,
        "stage2_sigma_min": 0.05,
        "stage2_peak_amp_init": 1.0,
        "stage3_center_init": 0.0,
        "stage3_one_hot_gamma": 0.284,
        "stage3_peak_amp_init": 1.0,
        "stage1_center_init_std": 0.5,
        "stage1_sigma_init": 1.0,
        "stage1_sigma_min": 0.05,
        "stage1_peak_amp_init": 1.0,
    }

    def __init__(self, config):
        config = dict(config)
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.device = torch.device(
            f"cuda:{self.gpu}" if torch.cuda.is_available() and self.use_gpu else "cpu"
        )
        self.data_root = os.path.join("dataset", self.data_path)

        self.train_loader = get_loader_segment(
            self.index,
            self.data_root,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            mode="train",
            dataset=self.dataset,
            num_workers=self.num_workers,
        )
        self.thre_loader = get_loader_segment(
            self.index,
            self.data_root,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            mode="thre",
            dataset=self.dataset,
            num_workers=self.num_workers,
        )
        self.test_loader = get_loader_segment(
            self.index,
            self.data_root,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            mode="test",
            dataset=self.dataset,
            num_workers=self.num_workers,
        )

        inferred_channels = int(self.train_loader.dataset.num_features)
        if self.input_c is None or int(self.input_c) <= 0:
            self.input_c = inferred_channels
        elif int(self.input_c) != inferred_channels:
            print(
                f"[Warning] input_c={self.input_c}, but dataset has {inferred_channels} channels. "
                f"Use dataset value."
            )
            self.input_c = inferred_channels
        if self.output_c is None or int(self.output_c) <= 0:
            self.output_c = self.input_c

        self.build_model()

    def build_model(self):
        self.model = GaussFuzzyAnomalyModel(
            window_size=self.win_size,
            num_features=self.input_c,
            stage2_num_patterns=self.stage2_num_patterns,
            stage1_center_init_std=self.stage1_center_init_std,
            stage1_sigma_init=self.stage1_sigma_init,
            stage1_sigma_min=self.stage1_sigma_min,
            stage1_peak_amp_init=self.stage1_peak_amp_init,
            stage2_center_init_std=self.stage2_center_init_std,
            stage2_sigma_init=self.stage2_sigma_init,
            stage2_sigma_min=self.stage2_sigma_min,
            stage3_center_init=self.stage3_center_init,
            stage3_one_hot_gamma=self.stage3_one_hot_gamma,
            stage3_peak_amp_init=self.stage3_peak_amp_init,
            stage2_peak_amp_init=self.stage2_peak_amp_init,
            top_q=self.top_q,
            score_type=self.score_type,
        ).to(self.device)

        self.stage1_optimizer = torch.optim.Adam(
            self.model.stage1_parameters(),
            lr=self.stage1_lr,
            weight_decay=self.stage1_weight_decay,
        )
        self.stage23_optimizer = torch.optim.Adam(
            self.model.stage23_parameters(),
            lr=self.stage23_lr,
        )

    def is_using_gpu(self):
        try:
            return bool(self.device.type == "cuda" and next(self.model.parameters()).is_cuda)
        except StopIteration:
            return bool(self.device.type == "cuda")

    def runtime_device_info(self):
        model_device = "unknown"
        try:
            model_device = str(next(self.model.parameters()).device)
        except StopIteration:
            pass

        gpu_name = "None"
        if torch.cuda.is_available() and self.is_using_gpu():
            gpu_index = int(self.device.index if self.device.index is not None else self.gpu)
            gpu_name = torch.cuda.get_device_name(gpu_index)

        return {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "use_gpu_arg": bool(self.use_gpu),
            "using_gpu": self.is_using_gpu(),
            "runtime_device": str(self.device),
            "model_device": model_device,
            "gpu_name": gpu_name,
        }

    def print_runtime_device_info(self):
        info = self.runtime_device_info()
        print("\n================ Runtime Device ================")
        print(f"torch.cuda.is_available : {info['torch_cuda_available']}")
        print(f"use_gpu argument       : {info['use_gpu_arg']}")
        print(f"Using GPU              : {info['using_gpu']}")
        print(f"Runtime device         : {info['runtime_device']}")
        print(f"Model device           : {info['model_device']}")
        print(f"GPU name               : {info['gpu_name']}")
        print("================================================")

    def _checkpoint_path(self) -> Path:
        save_name = self.save_name
        if save_name is None:
            save_name = f"{self.data_path}_FuzzyCD_L{self.win_size}_K{self.stage2_num_patterns}.pt"
        return Path(self.model_save_path) / save_name

    def _auto_stage1_path(self) -> Path:
        if self.save_name:
            stage1_name = f"{Path(str(self.save_name)).stem}_gaussian_direct_stage1.pt"
        else:
            stage1_name = (
                f"{self.data_path}_T{self.win_size}_K{self.stage2_num_patterns}"
                "_gaussian_direct_stage1.pt"
            )
        return (
            Path(self.model_save_path)
            / stage1_name
        )

    def _resolve_stage1_path(self) -> Path | None:
        if self.stage1_checkpoint is None:
            return None
        ckpt = str(self.stage1_checkpoint).strip()
        if ckpt.lower() == "none":
            return None
        if ckpt.lower() == "auto":
            path = self._auto_stage1_path()
            return path if path.exists() else None
        return Path(ckpt)

    def _try_load_stage1_reconstruction(self) -> bool:
        path = self._resolve_stage1_path()
        if path is None or not path.exists():
            return False

        checkpoint = torch_load_checkpoint(path, map_location="cpu")
        if int(checkpoint.get("window_size", self.win_size)) != int(self.win_size):
            print(f"[Stage 1] Skip checkpoint with mismatched window size: {path}")
            return False
        checkpoint_patterns = checkpoint.get("stage2_num_patterns", self.stage2_num_patterns)
        if int(checkpoint_patterns) != int(self.stage2_num_patterns):
            print(f"[Stage 1] Skip checkpoint with mismatched stage2_num_patterns: {path}")
            return False
        checkpoint_segments = checkpoint.get("input_segments", self.stage2_num_patterns + 1)
        if int(checkpoint_segments) != int(self.stage2_num_patterns + 1):
            print(f"[Stage 1] Skip checkpoint with mismatched input_segments: {path}")
            return False
        # Legacy checkpoints used input_segments = stage2_num_patterns + 2
        # because the mask itself was concatenated into the Stage 1 input.
        # Legacy code kept for reference:
        # checkpoint_segments = checkpoint.get("input_segments", self.stage2_num_patterns + 2)
        # if int(checkpoint_segments) != int(self.stage2_num_patterns + 2):
        #     print(f"[Stage 1] Skip checkpoint with mismatched input_segments: {path}")
        #     return False
        if "stage1_state" in checkpoint:
            try:
                self.model.stage1.load_state_dict(checkpoint["stage1_state"], strict=True)
                print(f"[Stage 1] Loaded Gaussian direct reconstruction checkpoint: {path}")
                return True
            except RuntimeError as exc:
                print(f"[Stage 1] Skip incompatible Gaussian direct checkpoint: {exc}")
                return False

        print(f"[Stage 1] Checkpoint has no Gaussian direct Stage 1 state; use random initialization: {path}")
        return False

    def _save_stage1_checkpoint(self):
        path = self._auto_stage1_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "stage1_state": self.model.stage1.state_dict(),
            "dataset": self.dataset,
            "data_path": self.data_path,
            "window_size": self.win_size,
            "input_c": self.input_c,
            "stage2_num_patterns": self.stage2_num_patterns,
            "input_segments": self.stage2_num_patterns + 1,
            # Legacy metadata kept for reference:
            # "input_segments": self.stage2_num_patterns + 2,
            "stage2_peak_amp_init": self.stage2_peak_amp_init,
            "stage3_center_init": self.stage3_center_init,
            "stage1_center_init_std": self.stage1_center_init_std,
            "stage1_sigma_init": self.stage1_sigma_init,
            "stage1_sigma_min": self.stage1_sigma_min,
            "stage1_peak_amp_init": self.stage1_peak_amp_init,
        }
        torch.save(payload, path)
        print(f"[Stage 1] Saved Gaussian direct reconstruction checkpoint: {path}")

    def _save_checkpoint(self, extra=None):
        path = self._checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        config_keys = [
            "win_size",
            "step",
            "anormly_ratio",
            "batch_size",
            "epochs",
            "stage23_lr",
            "dataset",
            "data_path",
            "input_c",
            "output_c",
            "stage1_lr",
            "stage1_weight_decay",
            "stage1_mask_ratio",
            "stage1_center_init_std",
            "stage1_sigma_init",
            "stage1_sigma_min",
            "stage1_peak_amp_init",
            "stage2_num_patterns",
            "top_q",
            "stage2_center_init_std",
            "stage2_sigma_init",
            "stage2_sigma_min",
            "stage2_peak_amp_init",
            "stage3_center_init",
            "stage3_one_hot_gamma",
            "stage3_peak_amp_init",
            "score_type",
            "threshold_source",
            "vus_sliding_window",
            "seed",
        ]
        safe_config = {key: getattr(self, key) for key in config_keys if hasattr(self, key)}
        payload = {
            "model_state": self.model.state_dict(),
            "dataset": self.dataset,
            "data_path": self.data_path,
            "win_size": self.win_size,
            "input_c": self.input_c,
            "stage2_num_patterns": self.stage2_num_patterns,
            "top_q": self.top_q,
            "config": safe_config,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[Checkpoint] Saved: {path}")

    def _load_checkpoint(self):
        path = self._checkpoint_path()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch_load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        print(f"[Checkpoint] Loaded: {path}")

    @staticmethod
    def _set_trainable(module, trainable: bool):
        for param in module.parameters():
            param.requires_grad_(trainable)

    def _set_stage1_trainable(self, trainable: bool):
        self._set_trainable(self.model.stage1, trainable)
        self.model.stage1.train(trainable)

    def _set_stage23_trainable(self, trainable: bool):
        self._set_trainable(self.model.stage2, trainable)
        self._set_trainable(self.model.stage3, trainable)
        self.model.stage2.train(trainable)
        self.model.stage3.train(trainable)

    def _mixed_mask(self, x):
        return make_random_mask(x, self.stage1_mask_ratio)

    def _make_view_masks(self, windows, force_endpoint=False):
        masks = []
        for d in range(self.input_c):
            x = windows[:, :, d]
            mask = make_random_mask(x, self.stage1_mask_ratio)
            if force_endpoint:
                mask[:, -1] = 0.0
            masks.append(mask)
        return torch.stack(masks, dim=-1)

    def _stage1_reconstruction_loss(self, windows):
        batch_size, window_size, num_features = windows.shape
        x = windows.permute(0, 2, 1).reshape(batch_size * num_features, window_size)
        mask = self._mixed_mask(x)
        masks = mask.reshape(batch_size, num_features, window_size).permute(0, 2, 1)
        rec_windows = self.model.build_reconstruction_view(windows, masks)
        x_hat = rec_windows.permute(0, 2, 1).reshape(
            batch_size * num_features, window_size
        )
        # Current GitHub version: reconstruction MSE is computed over every
        # point in the window, including both observed and masked positions.
        loss = torch.mean((x_hat - x).pow(2))

        # Legacy version kept for reference: MSE was computed only on masked
        # positions.
        # loss = masked_mse_loss(x_hat, x, mask)
        return loss, rec_windows

    def _train_stage1_epoch(self, epoch: int, total_epochs: int):
        self._set_stage1_trainable(True)
        self._set_stage23_trainable(False)
        epoch_time = time.time()
        total_loss = 0.0
        total_count = 0

        for input_data, _ in self.train_loader:
            windows = input_data.float().to(self.device)  # [B, L, D]
            self.stage1_optimizer.zero_grad(set_to_none=True)
            loss, _ = self._stage1_reconstruction_loss(windows)
            loss.backward()
            self.stage1_optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_count += 1

        print(
            f"Stage 1 Gaussian Direct Reconstruction Epoch {epoch:03d}/{total_epochs:03d} | "
            f"full_mse={total_loss / max(total_count, 1):.6f} | "
            f"time={time.time() - epoch_time:.2f}s"
        )

    def _stage23_membership_loss(self, windows, rec_windows=None, masks=None):
        if rec_windows is None:
            if masks is None:
                masks = self._make_view_masks(windows, force_endpoint=False)
            rec_windows = self.model.build_reconstruction_view(windows, masks)

        output = self.model.forward_dual(windows, rec_windows)
        u_raw = output["u_raw"]
        u_rec = output["u_rec"]

        mse_loss = torch.mean((u_raw - u_rec).pow(2))
        total = mse_loss
        parts = {
            "total": float(total.detach().cpu()),
            "mse_loss": float(mse_loss.detach().cpu()),
        }
        return total, parts

    def _train_stage23_epoch(self, epoch: int, total_epochs: int):
        self._set_stage1_trainable(False)
        self._set_stage23_trainable(True)
        epoch_time = time.time()
        sums = {
            "total": 0.0,
            "mse_loss": 0.0,
        }
        steps = 0

        for input_data, _ in self.train_loader:
            windows = input_data.float().to(self.device)
            self.stage23_optimizer.zero_grad(set_to_none=True)
            loss, parts = self._stage23_membership_loss(windows)
            loss.backward()
            self.stage23_optimizer.step()

            for key in sums:
                sums[key] += parts[key]
            steps += 1

        adjust_learning_rate(self.stage23_optimizer, epoch, self.stage23_lr)
        msg = " | ".join(f"{key}={sums[key] / max(steps, 1):.6f}" for key in sums)
        print(
            f"Stage 2 Gaussian Feature + Stage 3 Learnable Membership Epoch {epoch:03d}/{total_epochs:03d} | {msg} | "
            f"time={time.time() - epoch_time:.2f}s"
        )

    def _prepare_for_evaluation(self):
        self._set_stage1_trainable(False)
        self._set_stage23_trainable(False)
        self.model.eval()
        print("[Training] Stage 1, Stage 2, and Stage 3 are set to eval mode.")

    def train(self):
        self._try_load_stage1_reconstruction()
        total_rounds = int(self.epochs)
        print("\n================ Three-Stage Alternating Training ================")
        print(f"Epochs                 : {total_rounds}")
        print("Stage 1                : Gaussian direct reconstruction")
        print("Stage 2                : Gaussian feature extraction")
        print("Stage 3                : Learnable Gaussian membership over Stage 2 softmax one-hot distance")
        for round_idx in range(1, total_rounds + 1):
            print(f"\n[Alternating Epoch {round_idx:03d}/{total_rounds:03d}]")
            self._train_stage1_epoch(round_idx, total_rounds)
            self._train_stage23_epoch(round_idx, total_rounds)

        self._save_stage1_checkpoint()
        self._prepare_for_evaluation()
        self._save_checkpoint()

    def _score_loader(self, loader):
        self.model.eval()
        scores = []
        labels = []
        with torch.no_grad():
            for input_data, batch_labels in loader:
                windows = input_data.float().to(self.device)
                masks = self._make_view_masks(windows, force_endpoint=True)
                batch_scores, _ = self.model.anomaly_score(windows, masks)
                scores.append(batch_scores.detach().cpu().numpy())
                labels.append(self._endpoint_labels(batch_labels).numpy())

        return np.concatenate(scores, axis=0).reshape(-1), np.concatenate(labels, axis=0).reshape(-1)

    @staticmethod
    def _endpoint_labels(batch_labels):
        labels = torch.as_tensor(batch_labels)
        if labels.ndim == 3:
            labels = labels.max(dim=-1).values
        if labels.ndim == 2:
            labels = labels[:, -1]
        return labels.float()

    @staticmethod
    def _point_adjust(pred, gt):
        pred = pred.copy()
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        return pred

    def _compute_vus(self, gt, scores):
        from metrics.vus.utils.metrics import metricor

        labels = np.asarray(gt, dtype=int).reshape(-1)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if labels.size == 0 or scores.size == 0 or labels.size != scores.size:
            raise ValueError("VUS input labels and scores must be non-empty with the same length.")
        if np.sum(labels) == 0:
            raise ValueError("VUS requires at least one anomaly point in ground-truth labels.")
        window = int(getattr(self, "vus_sliding_window", 100))
        window = max(1, window)
        _, _, _, _, vus_roc, vus_pr = metricor().RangeAUC_volume(
            labels_original=labels,
            score=scores,
            windowSize=2 * window,
        )
        return float(vus_roc), float(vus_pr)

    def test(self):
        if self.mode == "test":
            self._load_checkpoint()

        print("\n==================== Test ====================")
        train_scores, _ = self._score_loader(self.train_loader)
        thre_scores, _ = self._score_loader(self.thre_loader)
        if self.threshold_source == "combined":
            threshold_base = np.concatenate([train_scores, thre_scores], axis=0)
        else:
            threshold_base = train_scores
        thresh = np.percentile(threshold_base, 100 - self.anormly_ratio)
        print("anormly_ratio", self.anormly_ratio)
        print("Threshold :", thresh)

        test_scores, gt = self._score_loader(self.test_loader)
        pred = (test_scores >= thresh).astype(int)
        gt = gt.astype(int)

        try:
            from metrics.combine_all_scores import combine_all_evaluation_scores

            scores_simple = combine_all_evaluation_scores(pred, gt, test_scores)
            metric_warnings = scores_simple.pop("_warnings", {})
            hidden_metrics = {"f1_score_ori", "f05_score_ori", "f1_score_pa", "f1_score_c"}
            for key, value in scores_simple.items():
                if key in hidden_metrics:
                    continue
                print("{0:21} : {1:0.4f}".format(key, value))
            for key, value in metric_warnings.items():
                print(f"[Metric warning] {key} skipped: {value}")
        except Exception as exc:
            print(f"[Metric warning] combine_all_evaluation_scores failed: {exc}")

        adjusted_pred = self._point_adjust(pred, gt)
        accuracy = accuracy_score(gt, adjusted_pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, adjusted_pred, average="binary", zero_division=0
        )
        result_dir = Path(self.result_path)
        result_dir.mkdir(parents=True, exist_ok=True)
        np.save(result_dir / f"{self.data_path}_scores.npy", test_scores)
        np.save(result_dir / f"{self.data_path}_pred.npy", adjusted_pred)
        np.save(result_dir / f"{self.data_path}_gt.npy", gt)

        return accuracy, precision, recall, f_score
