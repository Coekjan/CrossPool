"""Compute host-side numerical oracles for native debug loopback paths."""

import torch


def expected_loopback_rotation(hidden_states: torch.Tensor) -> torch.Tensor:
    """Return the pairwise rotation implemented by both loopback sites."""

    x_values = hidden_states.float()[..., 0::2]
    y_values = hidden_states.float()[..., 1::2]
    output = torch.empty_like(hidden_states.float())
    output[..., 0::2] = (x_values - y_values) / (2.0**0.5)
    output[..., 1::2] = (x_values + y_values) / (2.0**0.5)
    return output.to(hidden_states.dtype)
