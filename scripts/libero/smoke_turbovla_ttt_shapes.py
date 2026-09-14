#!/usr/bin/env python3
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from turbovla.models.turbovla import build_turbovla


def build_args(enable_ttt: bool, dinov3_path: str, bert_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        dinov3_path=dinov3_path,
        bert_path=bert_path,
        hidden_dim=256,
        nheads=8,
        dim_feedforward=2048,
        max_text_len=256,
        text_padding_length=21,
        text_padding_length_by_instruction={},
        vla_feature_enhancer_layers=1,
        enhancer_inner_dim=512,
        text_dropout=0.0,
        fusion_dropout=0.0,
        fusion_droppath=0.0,
        action_dim=7,
        chunk_size=12,
        state_dim=8,
        num_state_tokens=2,
        local_files_only=True,
        freeze_vision_encoder=True,
        freeze_text_encoder=True,
        dinov3_precision="fp32",
        num_views=2,
        image_size=256,
        position_embedding="view",
        encode_views_separately=True,
        padding_strategy="key_padding_mask",
        enable_ttt=enable_ttt,
        ttt_inner_lr_init=0.01,
        ttt_gate_init=1e-4,
        tbptt_step_size=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3_path", default="pretrained/dinov3-vitb16-from-meta")
    parser.add_argument("--bert_path", default="pretrained/bert-base-uncased")
    args = parser.parse_args()

    torch.manual_seed(0)
    instructions = ["put the bowl on the plate", "open the drawer"]
    images = torch.zeros(2, 2, 3, 256, 256)
    states = torch.zeros(2, 8)

    baseline = build_turbovla(build_args(False, args.dinov3_path, args.bert_path)).eval()
    with torch.no_grad():
        single = baseline(instructions, {"dinov3": images}, states)
        single_step = baseline.forward_step(instructions, {"dinov3": images}, states, return_ttt_memory=False)
    assert single.shape == (2, 12, 7)
    assert torch.allclose(single, single_step)

    ttt_model = build_turbovla(build_args(True, args.dinov3_path, args.bert_path)).eval()
    sequence_images = images[:, None].expand(2, 4, 2, 3, 256, 256).contiguous()
    sequence_states = states[:, None].expand(2, 4, 8).contiguous()
    sequence = ttt_model(instructions, {"dinov3": sequence_images}, sequence_states)
    assert sequence.shape == (2, 4, 12, 7)

    step0, mem0 = ttt_model.forward_step(instructions, {"dinov3": images}, states)
    step1, mem1 = ttt_model.forward_step(instructions, {"dinov3": images}, states, prev_ttt_memory=mem0)
    assert step0.shape == (2, 12, 7)
    assert step1.shape == (2, 12, 7)
    assert mem0.step == 1
    assert mem1.step == 2

    print("turbovla_ttt_shapes_smoke ok")
    print("single", tuple(single.shape))
    print("sequence", tuple(sequence.shape))
    print("memory_steps", mem0.step, mem1.step)
    print("ttt_stats", ttt_model.ttt.last_stats())


if __name__ == "__main__":
    main()
