#!/usr/bin/env python3
from __future__ import annotations

import torch

from turbovla.models.ttt import TemporalTTT


def main() -> None:
    torch.manual_seed(0)
    module = TemporalTTT(dim=256, inner_lr_init=0.01, gate_init=1e-4)
    x = torch.randn(2, 4, 13, 256)
    y, memory = module(x)
    assert y.shape == x.shape
    assert memory.step == 4

    first_step_out, first_memory = module.forward_step(x[:, 0])
    second_from_carry, second_memory = module.forward_step(x[:, 1], prev_memory=first_memory)
    second_from_reset, reset_memory = module.forward_step(x[:, 1])
    assert first_step_out.shape == x[:, 0].shape
    assert first_memory.step == 1
    assert second_memory.step == 2
    assert reset_memory.step == 1

    weights = next(iter(first_memory.fast_weights.values()))
    assert not torch.allclose(weights[0], weights[1])
    carried = next(iter(second_memory.fast_weights.values()))
    reset = next(iter(reset_memory.fast_weights.values()))
    assert not torch.allclose(carried, reset)

    identity_delta = (y - x).norm() / x.norm().clamp_min(1e-8)
    assert identity_delta.item() < 1e-3
    actual_lr = module.memory.actual_ttt_lr.item()
    assert abs(actual_lr - 0.01) < 1e-6

    loss = y.pow(2).mean()
    loss.backward()
    checked = {
        "raw_ttt_lr": module.memory.raw_ttt_lr,
        "ttt_gate": module.memory.ttt_gate,
        "to_qkv.weight": module.memory.to_qkv.weight,
        "fast_model.net.0.weight": module.memory.fast_model.net[0].weight,
    }
    for name, param in checked.items():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name

    print("ttt_core_smoke ok")
    print("shape", tuple(y.shape))
    print("actual_ttt_lr", actual_lr)
    print("identity_relative_delta", identity_delta.item())
    print("stats", module.last_stats())


if __name__ == "__main__":
    main()
