"""
Gromov-Wasserstein Structure Alignment Module
==============================================
The CORE contribution of RPGW-Net.

Replaces MMD-based alignment (DAGCN) with explicit graph structure alignment
via Gromov-Wasserstein distance.

Supports four GW variants:
  - vanilla GW:       standard GW distance
  - entropic GW:      GW + entropy regularization (EGW, faster + differentiable)
  - fused GW:         GW + feature cost trade-off (FGW)
  - partial GW:       only transport a fraction of mass (robust to noise)

Also implements GW_MultiInit strategy (Seyedi et al.) to escape local optima.

Dependency: POT (Python Optimal Transport) library
"""

import numpy as np
import ot
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class GWAlignment(nn.Module):
    """
    GW-based structure alignment between source and target graphs.

    Input:  (C_s, C_t) — pairwise distance matrices from GAT encoders
            (H_s, H_t) — node embeddings (for Fused GW)
    Output: alignment loss + aligned transport plan
    """

    def __init__(
        self,
        gw_type: str = "partial",
        epsilon: float = 0.8,
        alpha: float = 0.7,
        partial_mass: float = 0.85,
        multi_init: int = 5,
        reg: float = 1.0,
    ):
        """
        Args:
            gw_type:      GW variant: 'vanilla' | 'entropic' | 'fused' | 'partial'
            epsilon:      Entropy regularization strength (for EGW/FGW)
            alpha:        Feature vs structure trade-off (for FGW, 0=W, 1=GW)
            partial_mass: Transport mass fraction (for Partial GW, 0<m<=1)
            multi_init:   Number of random initializations (GW_MultiInit, 1 = single)
            reg:          Overall GW loss weight
        """
        super().__init__()
        self.gw_type = gw_type
        self.epsilon = epsilon
        self.alpha = alpha
        self.partial_mass = partial_mass
        self.multi_init = multi_init
        self.reg = reg

        # Small constant for numerical stability
        self.eps = 1e-12

    def forward(
        self,
        C_s: torch.Tensor,
        C_t: torch.Tensor,
        H_s: torch.Tensor = None,
        H_t: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Align source and target graph structures.

        Args:
            C_s:  (B, N, N) source domain distance matrix
            C_t:  (B, N, N) target domain distance matrix
            H_s:  (B, N, D) source node embeddings (for Fused GW)
            H_t:  (B, N, D) target node embeddings (for Fused GW)

        Returns:
            Dict with:
                'gw_loss':     scalar GW alignment loss
                'transport':   (B, N, N) optimal transport plan P
                'aligned_H_t': (B, N, D) target embeddings aligned to source
        """
        B, N, _ = C_s.shape
        device = C_s.device

        # Uniform marginals (equal weight per node)
        p = torch.ones(B, N, device=device) / N
        q = torch.ones(B, N, device=device) / N

        # Convert to numpy for POT (POT works on CPU numpy arrays)
        C_s_np = C_s.detach().cpu().numpy()
        C_t_np = C_t.detach().cpu().numpy()
        p_np = p.cpu().numpy()
        q_np = q.cpu().numpy()

        # Feature cost matrix M (for Fused GW)
        if self.gw_type == "fused" and H_s is not None and H_t is not None:
            M_np = torch.cdist(H_s, H_t, p=2).detach().cpu().numpy()  # (B, N, N)

        batch_gw_loss = []
        batch_transport = []

        for b in range(B):
            # --- Solve GW problem ---
            P, loss = self._solve_gw(
                C1=C_s_np[b],
                C2=C_t_np[b],
                p=p_np[b],
                q=q_np[b],
                M=M_np[b] if (self.gw_type == "fused" and H_s is not None) else None,
            )
            batch_gw_loss.append(loss)
            batch_transport.append(torch.from_numpy(P).float().to(device))

        gw_loss = self.reg * sum(batch_gw_loss) / B
        transport = torch.stack(batch_transport, dim=0)  # (B, N, N)

        # Align target embeddings: H_t_aligned = P^T · H_s (barycentric mapping)
        if H_s is not None:
            aligned_H_t = transport.transpose(1, 2) @ H_s  # (B, N, D)
        else:
            aligned_H_t = transport

        return {
            "gw_loss":       gw_loss,
            "transport":     transport,
            "aligned_H_t":   aligned_H_t,
        }

    def _solve_gw(
        self,
        C1: np.ndarray,
        C2: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
        M: np.ndarray = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Solve the GW problem for one sample.

        Returns:
            P:    Optimal transport plan (N × N)
            loss: GW cost value
        """
        best_loss = float("inf")
        best_P = None

        for trial in range(self.multi_init):
            # Generate initial transport plan
            if trial == 0:
                # Default: product of marginals
                P0 = p[:, None] * q[None, :]
            else:
                # Random initialization (projected to admissible set)
                P0 = np.random.rand(C1.shape[0], C2.shape[0]) + 1e-8
                # Sinkhorn projection to satisfy marginals (quick approx)
                P0 = self._sinkhorn_project(P0, p, q)

            try:
                if self.gw_type == "vanilla":
                    P = ot.gromov.gromov_wasserstein(C1, C2, p, q, G0=P0, max_iter=500)
                elif self.gw_type == "entropic":
                    P = ot.gromov.entropic_gromov_wasserstein(
                        C1, C2, p, q, epsilon=self.epsilon, G0=P0, max_iter=500
                    )
                elif self.gw_type == "fused":
                    # Fused GW: (1-α)·<M, P> + α·GW(C1, C2, P)
                    P = ot.gromov.fused_gromov_wasserstein(
                        M, C1, C2, p, q, alpha=self.alpha, G0=P0, max_iter=500
                    )
                elif self.gw_type == "partial":
                    P = ot.gromov.partial_gromov_wasserstein(
                        C1, C2, p, q, m=self.partial_mass, G0=P0, max_iter=500
                    )
                else:
                    raise ValueError(f"Unknown gw_type: {self.gw_type}")
            except Exception:
                # Numerical failure → skip this trial
                continue

            # Compute GW cost for this trial
            if self.gw_type == "fused" and M is not None:
                loss = self._gw_cost(C1, C2, P) * self.alpha + np.sum(M * P) * (1 - self.alpha)
            else:
                loss = self._gw_cost(C1, C2, P)

            if loss < best_loss:
                best_loss = loss
                best_P = P

        if best_P is None:
            # All trials failed → fallback to identity-like plan
            best_P = np.eye(C1.shape[0], C2.shape[0]) / C1.shape[0]
            best_loss = float("inf")

        return best_P, best_loss

    @staticmethod
    def _gw_cost(C1: np.ndarray, C2: np.ndarray, P: np.ndarray) -> float:
        """
        Compute the GW discrepancy:
            L_GW = Σ_{i,j,k,l} (C1[ij] - C2[kl])² · P[ik] · P[jl]
        """
        # Efficient computation via tensor operations
        C1_sq = C1 ** 2
        C2_sq = C2 ** 2

        t1 = np.dot(np.dot(C1_sq, P), np.ones(P.shape[1]))
        t1 = np.dot(t1, np.ones(P.shape[0]))

        t2 = np.dot(np.dot(np.ones(P.shape[0]), P), C2_sq.T)
        t2 = np.dot(t2, np.ones(P.shape[0]))

        t3 = -2 * np.trace(C1 @ P @ C2.T @ P.T)

        cost = t1 + t2 + t3
        return float(cost)

    @staticmethod
    def _sinkhorn_project(
        P0: np.ndarray, p: np.ndarray, q: np.ndarray, max_iter: int = 100, tol: float = 1e-9
    ) -> np.ndarray:
        """
        Project a matrix onto the coupling polytope Π(p, q) via Sinkhorn iterations.
        Used for generating valid random initializations.
        """
        P = P0.copy()
        for _ in range(max_iter):
            # Row scaling
            row_sum = P.sum(axis=1)
            row_sum[row_sum < 1e-12] = 1e-12
            P = P * (p / row_sum)[:, None]

            # Column scaling
            col_sum = P.sum(axis=0)
            col_sum[col_sum < 1e-12] = 1e-12
            P = P * (q / col_sum)[None, :]

            # Check convergence
            if np.max(np.abs(P.sum(axis=1) - p)) < tol and np.max(np.abs(P.sum(axis=0) - q)) < tol:
                break

        return P
