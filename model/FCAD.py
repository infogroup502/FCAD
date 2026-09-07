from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def inverse_softplus(value: float) -> float:
    value = float(value)
    return math.log(math.exp(value) - 1.0)


def make_random_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Create a random binary mask with at least one masked point per sample."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1).")

    mask = (torch.rand_like(x) > mask_ratio).float()
    no_missing = mask.sum(dim=-1) == x.shape[-1]
    if no_missing.any():
        rows = no_missing.nonzero(as_tuple=False).squeeze(-1)
        cols = torch.randint(0, x.shape[-1], (rows.numel(),), device=x.device)
        mask[rows, cols] = 0.0
    return mask


def masked_mse_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    missing = 1.0 - mask
    return ((x_hat - x).pow(2) * missing).sum() / missing.sum().clamp_min(eps)


class Stage1GaussianDirectReconstruction(nn.Module):

    def __init__(
        self,
        window_size: int,
        num_features: int,
        num_patterns: int,
        center_init_std: float = 0.5,
        sigma_init: float = 1.0,
        sigma_min: float = 0.05,
        peak_amp_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.num_features = int(num_features)
        self.num_patterns = int(num_patterns)
        self.input_segments = self.num_patterns + 1
        self.input_dim = self.input_segments * self.window_size
        self.output_dim = self.window_size
        self.sigma_min = float(sigma_min)

        centers = torch.randn(self.output_dim, self.input_dim)
        centers = centers * float(center_init_std)
        self.centers = nn.Parameter(centers)

        raw_sigma_value = inverse_softplus(max(float(sigma_init) - self.sigma_min, 1e-4))
        self.raw_sigma = nn.Parameter(
            torch.full((self.output_dim, self.input_dim), raw_sigma_value)
        )

        raw_peak_amp_value = inverse_softplus(max(float(peak_amp_init), 1e-4))
        self.raw_peak_amp = nn.Parameter(
            torch.full((self.output_dim, 1), raw_peak_amp_value, dtype=torch.float32)
        )

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + self.sigma_min

    @property
    def peak_amp(self) -> torch.Tensor:
        return F.softplus(self.raw_peak_amp) + 1e-6

    def _validate_windows_and_masks(self, windows: torch.Tensor, masks: torch.Tensor) -> None:
        if windows.ndim != 3:
            raise ValueError(f"windows must be [B, L, D], got {windows.shape}")
        if masks.shape != windows.shape:
            raise ValueError(f"masks shape {masks.shape} != windows shape {windows.shape}")
        if windows.shape[1] != self.window_size or windows.shape[2] != self.num_features:
            raise ValueError(
                f"Expected [B, {self.window_size}, {self.num_features}], got {windows.shape}"
            )

    def build_input(
        self,
        windows: torch.Tensor,
        masks: torch.Tensor,
        peak_density: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_windows_and_masks(windows, masks)
        expected_peak_shape = (self.num_features, self.num_patterns, self.window_size)
        if tuple(peak_density.shape) != expected_peak_shape:
            raise ValueError(
                f"Expected peak_density shape {expected_peak_shape}, got {tuple(peak_density.shape)}"
            )

        batch_size, window_size, num_features = windows.shape
        x = windows.permute(0, 2, 1).reshape(batch_size * num_features, window_size)
        mask = masks.permute(0, 2, 1).reshape(batch_size * num_features, window_size)

        peak_density = peak_density.to(device=windows.device, dtype=windows.dtype)
        peak_context = peak_density.unsqueeze(0).expand(batch_size, -1, -1, -1)
        peak_context = peak_context.reshape(
            batch_size * num_features,
            self.num_patterns * window_size,
        )
        # Current input: [x_masked, peak_density], length (K + 1) * L.
        return torch.cat([x * mask, peak_context], dim=-1)

        # return torch.cat([x * mask, mask, peak_context], dim=-1)

    def gaussian_direct(self, reconstruction_input: torch.Tensor) -> torch.Tensor:
        if reconstruction_input.ndim != 2:
            raise ValueError(
                f"reconstruction_input must be [N, {self.input_dim}], got {reconstruction_input.shape}"
            )
        if reconstruction_input.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {reconstruction_input.shape[-1]}"
            )

        z = reconstruction_input.unsqueeze(1) 
        centers = self.centers.unsqueeze(0)  
        sigma = self.sigma.unsqueeze(0)
        peak_amp = self.peak_amp.to(
            device=reconstruction_input.device,
            dtype=reconstruction_input.dtype,
        ).unsqueeze(0)  
        mu = peak_amp * torch.exp(-((z - centers).pow(2)) / (2.0 * sigma.pow(2)))
        return torch.sum(z * mu, dim=-1)  

    def forward(
        self,
        windows: torch.Tensor,
        masks: torch.Tensor,
        peak_density: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, window_size, num_features = windows.shape
        reconstruction_input = self.build_input(windows, masks, peak_density)
        x_hat = self.gaussian_direct(reconstruction_input)
        return x_hat.reshape(batch_size, num_features, window_size).permute(0, 2, 1)


class Stage2GaussianFeatureExtractor(nn.Module):

    def __init__(
        self,
        window_size: int,
        num_features: int,
        num_patterns: int,
        center_init_std: float = 0.5,
        sigma_init: float = 1.0,
        sigma_min: float = 0.05,
        peak_amp_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.num_features = int(num_features)
        self.num_patterns = int(num_patterns)
        self.sigma_min = float(sigma_min)
        self.gaussian_norm = 1.0 / math.sqrt(2.0 * math.pi)

        centers = torch.randn(self.num_features, self.num_patterns, self.window_size)
        centers = centers * float(center_init_std)
        self.centers = nn.Parameter(centers)

        raw_value = inverse_softplus(max(float(sigma_init) - self.sigma_min, 1e-4))
        self.raw_sigma = nn.Parameter(
            torch.full((self.num_features, self.num_patterns, self.window_size), raw_value)
        )

        raw_peak_amp_value = inverse_softplus(max(float(peak_amp_init), 1e-4))
        self.raw_peak_amp = nn.Parameter(
            torch.full(
                (self.num_features, self.num_patterns, 1),
                raw_peak_amp_value,
                dtype=torch.float32,
            )
        )

    @property
    def sigma(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma) + self.sigma_min

    @property
    def peak_amp(self) -> torch.Tensor:
        return F.softplus(self.raw_peak_amp) + 1e-6

    def peak_density(self) -> torch.Tensor:
        return self.peak_amp * self.gaussian_norm / self.sigma

    def forward(self, windows: torch.Tensor) -> dict[str, torch.Tensor]:
        if windows.ndim != 3:
            raise ValueError(f"windows must be [B, L, D], got {windows.shape}")
        if windows.shape[1] != self.window_size or windows.shape[2] != self.num_features:
            raise ValueError(
                f"Expected [B, {self.window_size}, {self.num_features}], got {windows.shape}"
            )

        x = windows.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, L]
        centers = self.centers.unsqueeze(0)  # [1, D, K, L]
        sigma = self.sigma.unsqueeze(0)
        peak_amp = self.peak_amp.to(device=windows.device, dtype=windows.dtype).unsqueeze(0)

        mu = peak_amp * torch.exp(-((x - centers).pow(2)) / (2.0 * sigma.pow(2)))
        h = torch.sum(x * mu, dim=-1)
        p = F.softmax(h, dim=-1)
        return {"mu": mu, "h": h, "p": p}


class Stage3OneHotOrthogonalMembership(nn.Module):

    def __init__(
        self,
        num_features: int,
        num_patterns: int,
        center_init: float = 0.0,
        gamma_init: float = 0.5,
        peak_amp_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_patterns = int(num_patterns)

        centers = torch.full(
            (self.num_features, self.num_patterns),
            float(center_init),
            dtype=torch.float32,
        )
        self.centers = nn.Parameter(centers)

        one_hot_raw_gamma_value = inverse_softplus(max(float(gamma_init), 1e-4))
        one_hot_raw_gamma = torch.full(
            (self.num_features, self.num_patterns),
            one_hot_raw_gamma_value,
            dtype=torch.float32,
        )
        self.one_hot_raw_gamma = nn.Parameter(one_hot_raw_gamma)

        raw_peak_amp_value = inverse_softplus(max(float(peak_amp_init), 1e-4))
        self.raw_peak_amp = nn.Parameter(
            torch.full(
                (self.num_features, self.num_patterns),
                raw_peak_amp_value,
                dtype=torch.float32,
            )
        )

        self.register_buffer("one_hot", torch.eye(self.num_patterns, dtype=torch.float32))

    @property
    def one_hot_gamma(self) -> torch.Tensor:
        return F.softplus(self.one_hot_raw_gamma) + 1e-6

    @property
    def center(self) -> torch.Tensor:
        return self.centers

    @property
    def peak_amp(self) -> torch.Tensor:
        return F.softplus(self.raw_peak_amp) + 1e-6

    def forward(self, pattern_probs: torch.Tensor) -> dict[str, torch.Tensor]:
        if pattern_probs.ndim != 3:
            raise ValueError(f"pattern_probs must be [B, D, K], got {pattern_probs.shape}")
        if pattern_probs.shape[1] != self.num_features or pattern_probs.shape[2] != self.num_patterns:
            raise ValueError(
                f"Expected [B, {self.num_features}, {self.num_patterns}], got {pattern_probs.shape}"
            )

        p = pattern_probs
        anchors = self.one_hot.to(device=pattern_probs.device, dtype=pattern_probs.dtype)
        dist = torch.sum(
            (
                p.unsqueeze(-2)
                - anchors.view(1, 1, self.num_patterns, self.num_patterns)
            ).pow(2),
            dim=-1,
        )
        one_hot_gamma = self.one_hot_gamma.to(
            device=pattern_probs.device,
            dtype=pattern_probs.dtype,
        ).view(1, self.num_features, self.num_patterns)
        center = self.center.to(
            device=pattern_probs.device,
            dtype=pattern_probs.dtype,
        ).view(1, self.num_features, self.num_patterns)
        base_u = torch.exp(-((dist - center).pow(2)) / (2.0 * one_hot_gamma.pow(2)))
        peak_amp = self.peak_amp.to(
            device=pattern_probs.device,
            dtype=pattern_probs.dtype,
        ).view(1, self.num_features, self.num_patterns)
        u = peak_amp * base_u
        return {"p": p, "dist": dist, "u": u}


class GaussFuzzyAnomalyModel(nn.Module):
    def __init__(
        self,
        window_size: int,
        num_features: int,
        stage2_num_patterns: int,
        stage1_center_init_std: float,
        stage1_sigma_init: float,
        stage1_sigma_min: float,
        stage1_peak_amp_init: float,
        stage2_center_init_std: float,
        stage2_sigma_init: float,
        stage2_sigma_min: float,
        stage3_center_init: float,
        stage3_one_hot_gamma: float,
        stage3_peak_amp_init: float,
        stage2_peak_amp_init: float,
        top_q: int,
        score_type: str = "abs",
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.num_features = int(num_features)
        self.stage2_num_patterns = int(stage2_num_patterns)
        self.top_q = int(top_q)
        self.score_type = score_type

        self.stage1 = Stage1GaussianDirectReconstruction(
            window_size=window_size,
            num_features=num_features,
            num_patterns=stage2_num_patterns,
            center_init_std=stage1_center_init_std,
            sigma_init=stage1_sigma_init,
            sigma_min=stage1_sigma_min,
            peak_amp_init=stage1_peak_amp_init,
        )
        self.stage2 = Stage2GaussianFeatureExtractor(
            window_size=window_size,
            num_features=num_features,
            num_patterns=stage2_num_patterns,
            center_init_std=stage2_center_init_std,
            sigma_init=stage2_sigma_init,
            sigma_min=stage2_sigma_min,
            peak_amp_init=stage2_peak_amp_init,
        )
        self.stage3 = Stage3OneHotOrthogonalMembership(
            num_features=num_features,
            num_patterns=stage2_num_patterns,
            center_init=stage3_center_init,
            gamma_init=stage3_one_hot_gamma,
            peak_amp_init=stage3_peak_amp_init,
        )

    def stage1_parameters(self):
        yield from self.stage1.parameters()

    def stage2_parameters(self):
        yield from self.stage2.parameters()

    def stage3_parameters(self):
        yield from self.stage3.parameters()

    def stage23_parameters(self):
        yield from self.stage2.parameters()
        yield from self.stage3.parameters()

    def run_stage1(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        peak_density = self.stage2.peak_density()
        return self.stage1(windows, masks, peak_density)

    def run_stage2(self, windows: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.stage2(windows)

    def run_stage3(self, pattern_probs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.stage3(pattern_probs)

    def build_reconstruction_input(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        peak_density = self.stage2.peak_density()
        return self.stage1.build_input(windows, masks, peak_density)

    def build_reconstruction_view(self, windows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.run_stage1(windows, masks)

    def forward_dual(
        self,
        raw_windows: torch.Tensor,
        rec_windows: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if rec_windows is None:
            raise ValueError("rec_windows must be provided by build_reconstruction_view(...).")

        raw_stage2 = self.run_stage2(raw_windows)
        raw_stage3 = self.run_stage3(raw_stage2["p"])
        rec_stage2 = self.run_stage2(rec_windows)
        rec_stage3 = self.run_stage3(rec_stage2["p"])
        return {
            "mu_raw": raw_stage2["mu"],
            "h_raw": raw_stage2["h"],
            "p_raw": raw_stage3["p"],
            "dist_raw": raw_stage3["dist"],
            "u_raw": raw_stage3["u"],
            "mu_rec": rec_stage2["mu"],
            "h_rec": rec_stage2["h"],
            "p_rec": rec_stage3["p"],
            "dist_rec": rec_stage3["dist"],
            "u_rec": rec_stage3["u"],
            "rec_windows": rec_windows,
        }

    def anomaly_score(
        self,
        windows: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rec_windows = self.build_reconstruction_view(windows, masks)
        output = self.forward_dual(windows, rec_windows)
        raw_membership = output["u_raw"]
        rec_membership = output["u_rec"]
        if self.score_type == "square":
            diff = (raw_membership - rec_membership).pow(2)
        else:
            diff = torch.abs(raw_membership - rec_membership)

        variable_scores = diff.mean(dim=-1)  # [B, D]
        q = min(max(self.top_q, 1), self.num_features)
        top_scores = torch.topk(variable_scores, k=q, dim=-1).values
        final_score = top_scores.mean(dim=-1)
        output["variable_scores"] = variable_scores
        output["score"] = final_score
        return final_score, output
