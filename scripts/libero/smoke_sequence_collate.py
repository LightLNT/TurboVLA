#!/usr/bin/env python3
from __future__ import annotations

import torch

from turbovla.data.libero_rlds import vla_sequence_collate_fn


def make_step(value: float):
    img1 = {"dinov3": torch.full((3, 256, 256), value)}
    img2 = {"dinov3": torch.full((3, 256, 256), value + 0.5)}
    return img1, img2


def main() -> None:
    batch = []
    for b in range(2):
        images = [make_step(float(b * 10 + t)) for t in range(4)]
        instruction = f"instruction {b}"
        states = torch.randn(4, 8)
        actions = torch.randn(4, 12, 7)
        masks = torch.ones(4, 12)
        batch.append((images, instruction, states, actions, masks))

    samples, instructions, states, actions, masks = vla_sequence_collate_fn(batch)
    assert samples["dinov3"].shape == (2, 4, 2, 3, 256, 256)
    assert states.shape == (2, 4, 8)
    assert actions.shape == (2, 4, 12, 7)
    assert masks.shape == (2, 4, 12)
    assert instructions == ["instruction 0", "instruction 1"]
    assert samples["dinov3"][0, 0, 0, 0, 0, 0].item() == 0.0
    assert samples["dinov3"][0, 1, 0, 0, 0, 0].item() == 1.0
    assert samples["dinov3"][1, 3, 1, 0, 0, 0].item() == 13.5
    print("sequence_collate_smoke ok")
    print("samples[dinov3]", tuple(samples["dinov3"].shape))
    print("states", tuple(states.shape))
    print("actions", tuple(actions.shape))
    print("masks", tuple(masks.shape))


if __name__ == "__main__":
    main()
