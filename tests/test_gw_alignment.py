"""Unit tests for GW Alignment module."""

import torch
import numpy as np
from rpgw.models.gw_alignment import GWAlignment


def test_all_gw_variants():
    """Test that all 4 GW variants run without errors."""
    N = 32
    X_s = np.random.randn(N, 64)
    X_t = np.random.randn(N, 64) * 0.8 + 0.2

    C_s = torch.from_numpy(
        np.sqrt(((X_s[:, None] - X_s[None, :]) ** 2).sum(-1))
    ).float().unsqueeze(0)
    C_t = torch.from_numpy(
        np.sqrt(((X_t[:, None] - X_t[None, :]) ** 2).sum(-1))
    ).float().unsqueeze(0)
    H_s = torch.from_numpy(X_s).float().unsqueeze(0)
    H_t = torch.from_numpy(X_t).float().unsqueeze(0)

    for gw_type in ["vanilla", "entropic", "fused", "partial"]:
        gw = GWAlignment(gw_type=gw_type, multi_init=2)
        result = gw(C_s, C_t, H_s, H_t)

        assert isinstance(result["gw_loss"], torch.Tensor)
        assert result["transport"].shape == (1, N, N)
        assert result["aligned_H_t"].shape == (1, N, 64)


def test_multi_init_helps():
    """Test that multi-init gives <= cost than single init."""
    N = 16
    np.random.seed(123)
    X_s = np.random.randn(N, 10)
    X_t = np.random.randn(N, 10)

    C_s = torch.from_numpy(
        np.sqrt(((X_s[:, None] - X_s[None, :]) ** 2).sum(-1))
    ).float().unsqueeze(0)
    C_t = torch.from_numpy(
        np.sqrt(((X_t[:, None] - X_t[None, :]) ** 2).sum(-1))
    ).float().unsqueeze(0)

    gw_single = GWAlignment(gw_type="vanilla", multi_init=1)
    gw_multi = GWAlignment(gw_type="vanilla", multi_init=5)

    result_single = gw_single(C_s, C_t)
    result_multi = gw_multi(C_s, C_t)

    # Multi-init should NOT be worse (<= single init cost)
    assert result_multi["gw_loss"] <= result_single["gw_loss"] + 1e-5


if __name__ == "__main__":
    test_all_gw_variants()
    test_multi_init_helps()
    print("All GW tests passed!")
