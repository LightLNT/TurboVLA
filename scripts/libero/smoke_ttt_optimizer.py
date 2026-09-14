#!/usr/bin/env python3
from __future__ import annotations

import argparse
from types import SimpleNamespace

from turbovla.models.turbovla import build_turbovla
from turbovla.training.trainer import apply_requested_freezes, build_param_group_optimizer, freeze_backbones


def make_args(dinov3_path: str, bert_path: str) -> SimpleNamespace:
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
        enable_ttt=True,
        ttt_position="post_vl_fusion",
        ttt_inner_lr_init=0.01,
        ttt_gate_init=1e-4,
        tbptt_step_size=None,
        lr=1e-5,
        head_lr=1e-5,
        ttt_outer_lr=5e-5,
        dinov3_lr=5e-5,
        weight_decay=1e-10,
        head_weight_decay=1e-10,
        dinov3_weight_decay=1e-10,
        freeze_backbones=True,
        freeze_text_projection=True,
        freeze_vision_projection=True,
        freeze_vl_interaction=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov3_path", default="pretrained/dinov3-vitb16-from-meta")
    parser.add_argument("--bert_path", default="pretrained/bert-base-uncased")
    cli = parser.parse_args()

    args = make_args(cli.dinov3_path, cli.bert_path)
    model = build_turbovla(args)
    freeze_backbones(model)
    apply_requested_freezes(model, args)
    optimizer, summary = build_param_group_optimizer(model, args)
    del optimizer

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    ttt_trainable = sum(param.numel() for name, param in model.named_parameters() if name.startswith("ttt.") and param.requires_grad)
    action_trainable = sum(
        param.numel() for name, param in model.named_parameters() if name.startswith("action_head.") and param.requires_grad
    )
    assert ttt_trainable > 0
    assert action_trainable > 0
    assert any(group["name"].startswith("ttt_") for group in summary)
    print("ttt_optimizer_smoke ok")
    print("trainable_params", trainable)
    print("ttt_trainable_params", ttt_trainable)
    print("action_head_trainable_params", action_trainable)
    print("optimizer_groups", summary)


if __name__ == "__main__":
    main()
