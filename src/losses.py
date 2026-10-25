"""HLCC penalty and the annealed MSE + HLCC training loss."""
import math

import torch
import torch.nn as nn

from src.hlcc import moment_operator, residual_projectors


class HelgasonLudwigLoss(nn.Module):
    """HLCC penalty of orders 0..max_order for a parallel-beam sinogram.

    The raw loss of order n is the mean squared part of the moment curve a_n outside its harmonic space H_n
    (src/hlcc.py): L_n = ||(I - P_n) a_n||^2 / V. Order 0 is the variance of the mass over the views, order 1 the
    part of the first moment that is not A cos(theta) + B sin(theta). ``calibrate`` normalises every order.

    Input: [V, D] or [B, V, D].
    """

    moment_op: torch.Tensor
    residual_projectors: torch.Tensor
    scales: torch.Tensor

    def __init__(self, geom, max_order: int = 1):
        super().__init__()
        if max_order < 0:
            raise ValueError(f"max_order must be >= 0, got {max_order}")
        self.max_order = max_order
        self.num_views = geom.num_views
        self.half_width = geom.det_col_count * geom.det_pixel_size / 2.0

        self.register_buffer("moment_op", moment_operator(geom, max_order).float())                      # [D, K+1]
        self.register_buffer("residual_projectors", residual_projectors(geom.angles, max_order).float())  # [K+1, V, V]
        self.register_buffer("scales", torch.ones(max_order + 1))

    @torch.no_grad()
    def calibrate(self, sinogram: torch.Tensor) -> None:
        """Use the raw losses of ``sinogram`` as unit scales.

        Calibrate on the zero-filled inputs, not on the ground truth (whose loss is ~0): the normalised loss is
        then 1 for zero filling and ~0 for a consistent sinogram.
        """
        self.scales.copy_(torch.clamp(self.raw_losses(sinogram).detach(), min=1e-12))

    def moments(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Moment curves a_0..a_K: [B, V, K+1]."""
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)
        return sinogram @ self.moment_op

    def _residuals(self, sinogram: torch.Tensor) -> torch.Tensor:
        """(I - P_n) a_n: [B, K+1, V]."""
        return torch.einsum("nuv,bvn->bnu", self.residual_projectors, self.moments(sinogram))

    def raw_losses(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Unnormalised loss of every order, averaged over the batch: [K+1]."""
        return (self._residuals(sinogram) ** 2).mean(dim=(0, 2))

    @torch.no_grad()
    def relative_residuals(self, sinogram: torch.Tensor) -> torch.Tensor:
        """||(I - P_n) a_n||^2 / ||a_n||^2 per sample and order, [B, K+1]. Scale-free; float64 because the
        ground truth is around 1e-6."""
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)
        moments = sinogram.double() @ self.moment_op.double()                     # [B, V, K+1]
        residuals = torch.einsum("nuv,bvn->bnu", self.residual_projectors.double(), moments)
        return (residuals**2).sum(dim=2) / (moments**2).sum(dim=1).clamp_min(1e-30)

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        return self.raw_losses(sinogram) / self.scales


class AnnealedLoss(nn.Module):
    """total = MSE + alpha(epoch) * sum_n lambda_n * L_n, alpha going from 0 to 1 over ``anneal_epochs``.

    lambda_0 = ``lambda_m0``, lambda_1 = ``lambda_m1`` and every order n >= 2 shares ``lambda_high``.
    Only the HLCC terms are normalised (~1 on the zero-filled input); the MSE is ~2e-2 there, so the physics
    term weighs about 50 lambda times the data term. On the U-Net, lambda = 0.1 gives a physics gradient 26 to 52
    times the MSE one and the model under-fits, hence the default 1e-3. The ground truth satisfies the HLCC, so
    the penalty does not move the optimum: at best it regularises the optimisation path.
    """

    lambdas: torch.Tensor

    def __init__(self, geom, lambda_m0: float = 1e-3, lambda_m1: float = 1e-3, anneal_epochs: int = 20,
                 lambda_high: float = 1e-3, max_order: int = 1):
        super().__init__()
        self.data_loss_fn = nn.MSELoss()
        self.physics_loss_fn = HelgasonLudwigLoss(geom, max_order)
        lambdas = ([lambda_m0, lambda_m1] + [lambda_high] * max(0, max_order - 1))[: max_order + 1]
        self.register_buffer("lambdas", torch.tensor(lambdas, dtype=torch.float32))
        self.anneal_epochs = anneal_epochs

    def calibrate(self, sinogram: torch.Tensor) -> None:
        self.physics_loss_fn.calibrate(sinogram)

    def alpha(self, completed_epochs: int) -> float:
        """Cosine warm-up: 0 during the first epoch, 1 after ``anneal_epochs``."""
        if self.anneal_epochs <= 0:
            return 1.0
        t = min(1.0, max(0.0, completed_epochs / self.anneal_epochs))
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def forward(self, pred_sino: torch.Tensor, true_sino: torch.Tensor, current_epoch: int):
        """``current_epoch`` counts the completed epochs. Returns (total, MSE, normalised HLCC losses [K+1])."""
        loss_data = self.data_loss_fn(pred_sino, true_sino)
        physics_losses = self.physics_loss_fn(pred_sino)
        total_loss = loss_data + self.alpha(current_epoch) * (self.lambdas * physics_losses).sum()
        return total_loss, loss_data, physics_losses
