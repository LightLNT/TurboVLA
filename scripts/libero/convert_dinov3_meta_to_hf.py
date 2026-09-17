#!/usr/bin/env python3
"""Convert a Meta DINOv3 ViT checkpoint to a local Transformers directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessorFast, DINOv3ViTModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the Meta DINOv3 .pth state_dict.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the Transformers model files will be written.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


def infer_vitb_config(meta_state: dict[str, torch.Tensor], image_size: int) -> DINOv3ViTConfig:
    hidden_size = int(meta_state["cls_token"].shape[-1])
    num_hidden_layers = 1 + max(
        int(key.split(".")[1])
        for key in meta_state
        if key.startswith("blocks.") and key.endswith(".norm1.weight")
    )
    num_attention_heads = hidden_size // 64
    intermediate_size = int(meta_state["blocks.0.mlp.fc1.weight"].shape[0])
    patch_size = int(meta_state["patch_embed.proj.weight"].shape[-1])
    num_register_tokens = int(meta_state["storage_tokens"].shape[1])

    return DINOv3ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_register_tokens=num_register_tokens,
        layer_norm_eps=1e-5,
        rope_theta=100.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        use_gated_mlp=False,
    )


def convert_state_dict(meta_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {
        "embeddings.cls_token": meta_state["cls_token"],
        "embeddings.mask_token": meta_state["mask_token"].reshape(1, 1, -1),
        "embeddings.register_tokens": meta_state["storage_tokens"],
        "embeddings.patch_embeddings.weight": meta_state["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.bias": meta_state["patch_embed.proj.bias"],
        "norm.weight": meta_state["norm.weight"],
        "norm.bias": meta_state["norm.bias"],
    }

    hidden_size = converted["embeddings.cls_token"].shape[-1]
    num_layers = 1 + max(
        int(key.split(".")[1])
        for key in meta_state
        if key.startswith("blocks.") and key.endswith(".norm1.weight")
    )

    for layer_idx in range(num_layers):
        src = f"blocks.{layer_idx}"
        dst = f"layer.{layer_idx}"

        converted[f"{dst}.norm1.weight"] = meta_state[f"{src}.norm1.weight"]
        converted[f"{dst}.norm1.bias"] = meta_state[f"{src}.norm1.bias"]
        converted[f"{dst}.norm2.weight"] = meta_state[f"{src}.norm2.weight"]
        converted[f"{dst}.norm2.bias"] = meta_state[f"{src}.norm2.bias"]

        qkv_weight = meta_state[f"{src}.attn.qkv.weight"]
        q_weight, k_weight, v_weight = qkv_weight.split(hidden_size, dim=0)
        converted[f"{dst}.attention.q_proj.weight"] = q_weight
        converted[f"{dst}.attention.k_proj.weight"] = k_weight
        converted[f"{dst}.attention.v_proj.weight"] = v_weight

        qkv_bias = meta_state[f"{src}.attn.qkv.bias"]
        q_bias, _, v_bias = qkv_bias.split(hidden_size, dim=0)
        converted[f"{dst}.attention.q_proj.bias"] = q_bias
        converted[f"{dst}.attention.v_proj.bias"] = v_bias

        converted[f"{dst}.attention.o_proj.weight"] = meta_state[f"{src}.attn.proj.weight"]
        converted[f"{dst}.attention.o_proj.bias"] = meta_state[f"{src}.attn.proj.bias"]
        converted[f"{dst}.layer_scale1.lambda1"] = meta_state[f"{src}.ls1.gamma"]
        converted[f"{dst}.layer_scale2.lambda1"] = meta_state[f"{src}.ls2.gamma"]

        converted[f"{dst}.mlp.up_proj.weight"] = meta_state[f"{src}.mlp.fc1.weight"]
        converted[f"{dst}.mlp.up_proj.bias"] = meta_state[f"{src}.mlp.fc1.bias"]
        converted[f"{dst}.mlp.down_proj.weight"] = meta_state[f"{src}.mlp.fc2.weight"]
        converted[f"{dst}.mlp.down_proj.bias"] = meta_state[f"{src}.mlp.fc2.bias"]

    return converted


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta_state = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(meta_state, dict):
        raise TypeError(f"expected a state_dict, got {type(meta_state)!r}")

    config = infer_vitb_config(meta_state, image_size=args.image_size)
    model = DINOv3ViTModel(config)
    missing, unexpected = model.load_state_dict(convert_state_dict(meta_state), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"load_state_dict mismatch: missing={missing}, unexpected={unexpected}")

    model.save_pretrained(args.output_dir, safe_serialization=True)

    processor = DINOv3ViTImageProcessorFast(
        do_resize=True,
        size={"height": args.image_size, "width": args.image_size},
        do_rescale=True,
        rescale_factor=1.0 / 255.0,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )
    processor.save_pretrained(args.output_dir)

    print(f"saved DINOv3 HF model to {args.output_dir}")


if __name__ == "__main__":
    main()
