from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse_softplus value must be positive")
    if value < 20.0:
        return math.log(math.expm1(value))
    return value


@dataclass
class TTTMemory:
    fast_weights: dict[str, torch.Tensor]
    step: int = 0


class FastWeightMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MemoryKeyValueBind(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        inner_lr_init: float = 1e-2,
        gate_init: float = 1e-4,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.fast_model = FastWeightMLP(dim)
        self.raw_ttt_lr = nn.Parameter(torch.tensor(inverse_softplus(float(inner_lr_init))))
        self.ttt_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self._last_stats: dict[str, float] = {}

    @property
    def actual_ttt_lr(self) -> torch.Tensor:
        return F.softplus(self.raw_ttt_lr)

    def initial_memory(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> TTTMemory:
        fast_weights = {
            name: param.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, *param.shape)
            for name, param in self.fast_model.named_parameters()
        }
        return TTTMemory(fast_weights=fast_weights, step=0)

    def _retrieve(self, params: dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
        return functional_call(self.fast_model, params, (inputs,))

    def _loss(self, params: dict[str, torch.Tensor], keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        pred = self._retrieve(params, keys)
        return F.mse_loss(pred, values)

    @staticmethod
    def _tree_norm(params: dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for value in params.values():
            norm_sq = value.float().pow(2).sum()
            total = norm_sq if total is None else total + norm_sq
        if total is None:
            raise ValueError("cannot compute norm of empty parameter tree")
        return total.sqrt()

    def forward(
        self,
        tokens: torch.Tensor,
        prev_memory: TTTMemory | None = None,
    ) -> tuple[torch.Tensor, TTTMemory, dict[str, torch.Tensor]]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
        batch_size = tokens.shape[0]
        if prev_memory is None:
            memory = self.initial_memory(batch_size, device=tokens.device, dtype=tokens.dtype)
        else:
            memory = prev_memory

        q, k, v = self.to_qkv(tokens).chunk(3, dim=-1)
        grad_fn = grad(self._loss)
        grads = vmap(grad_fn, in_dims=(0, 0, 0))(memory.fast_weights, k, v)
        lr = self.actual_ttt_lr.to(device=tokens.device, dtype=tokens.dtype)
        next_fast_weights = {
            name: memory.fast_weights[name] - lr * grads[name]
            for name in memory.fast_weights
        }
        memory_out = vmap(self._retrieve, in_dims=(0, 0))(next_fast_weights, q)
        next_memory = TTTMemory(fast_weights=next_fast_weights, step=memory.step + 1)

        w_norm = self._tree_norm(memory.fast_weights)
        delta = {name: next_fast_weights[name] - memory.fast_weights[name] for name in memory.fast_weights}
        delta_norm = self._tree_norm(delta)
        stats_tensors = {
            "actual_ttt_lr": self.actual_ttt_lr.detach(),
            "ttt_gate_tanh": self.ttt_gate.detach().tanh(),
            "ttt_gate_abs": self.ttt_gate.detach().tanh().abs(),
            "fast_weight_norm": w_norm.detach(),
            "delta_fast_weight_norm": delta_norm.detach(),
            "relative_update_norm": (delta_norm / (w_norm + 1e-8)).detach(),
        }
        self._last_stats = {key: float(value.float().cpu()) for key, value in stats_tensors.items()}
        return memory_out, next_memory, stats_tensors

    def last_stats(self) -> dict[str, float]:
        return dict(self._last_stats)


class TemporalTTT(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        inner_lr_init: float = 1e-2,
        gate_init: float = 1e-4,
        tbptt_step_size: int | None = None,
    ) -> None:
        super().__init__()
        self.memory = MemoryKeyValueBind(dim, inner_lr_init=inner_lr_init, gate_init=gate_init)
        self.tbptt_step_size = tbptt_step_size
        self._last_stats: dict[str, float] = {}

    def forward(
        self,
        x: torch.Tensor,
        prev_memory: TTTMemory | None = None,
    ) -> tuple[torch.Tensor, TTTMemory]:
        if x.ndim != 4:
            raise ValueError(f"TTT input must be [B,T,N,D], got {tuple(x.shape)}")
        outputs = []
        curr_memory = prev_memory
        per_step_stats: list[dict[str, torch.Tensor]] = []
        gate = self.memory.ttt_gate.tanh().to(device=x.device, dtype=x.dtype)
        for step_idx in range(x.shape[1]):
            memory_out, curr_memory, stats = self.memory(x[:, step_idx], curr_memory)
            outputs.append(x[:, step_idx] + gate * memory_out)
            per_step_stats.append(stats)
            if self.tbptt_step_size is not None and (step_idx + 1) % self.tbptt_step_size == 0:
                curr_memory = TTTMemory(
                    fast_weights={name: value.detach() for name, value in curr_memory.fast_weights.items()},
                    step=curr_memory.step,
                )
        self._last_stats = self._average_stats(per_step_stats)
        return torch.stack(outputs, dim=1), curr_memory

    def forward_step(
        self,
        x: torch.Tensor,
        prev_memory: TTTMemory | None = None,
    ) -> tuple[torch.Tensor, TTTMemory]:
        if x.ndim != 3:
            raise ValueError(f"TTT step input must be [B,N,D], got {tuple(x.shape)}")
        y, next_memory = self.forward(x.unsqueeze(1), prev_memory=prev_memory)
        return y[:, 0], next_memory

    @staticmethod
    def _average_stats(per_step_stats: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        if not per_step_stats:
            return {}
        keys = per_step_stats[0].keys()
        return {
            key: float(torch.stack([stats[key].float() for stats in per_step_stats]).mean().detach().cpu())
            for key in keys
        }

    def last_stats(self) -> dict[str, float]:
        return dict(self._last_stats)


TTTWrapper = TemporalTTT
