#!/usr/bin/env python3
"""Filter no-op LIBERO transitions without dropping replay-failed demos.

This is a diagnostic alternative to regenerate_libero_no_noops.py. It keeps all
raw demonstrations, filters only action-level no-ops, and resizes raw 128px
images to the 256px resolution expected by the TurboVLA dataloader.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from scripts.libero.regenerate_libero_no_noops import is_noop


IMAGE_RESOLUTION = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero_raw_data_dir", required=True)
    parser.add_argument("--libero_target_dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _demo_sort_key(name: str) -> tuple[int, str]:
    suffix = name.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else -1, name)


def _resize_rgb_batch(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] uint8 images, got {images.shape}")
    if images.shape[1:3] == (IMAGE_RESOLUTION, IMAGE_RESOLUTION):
        return images.astype(np.uint8, copy=False)

    out = np.empty((images.shape[0], IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3), dtype=np.uint8)
    for i, frame in enumerate(images):
        out[i] = np.asarray(
            Image.fromarray(frame.astype(np.uint8, copy=False)).resize(
                (IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
    return out


def _filtered_indices(actions: np.ndarray) -> list[int]:
    keep: list[int] = []
    kept_actions: list[np.ndarray] = []
    for idx, action in enumerate(actions):
        prev_action = kept_actions[-1] if kept_actions else None
        if is_noop(action, prev_action):
            continue
        keep.append(idx)
        kept_actions.append(action)
    return keep


def _copy_obs_dataset(obs_in: h5py.Group, obs_out: h5py.Group, key: str, keep: list[int]) -> None:
    data = np.asarray(obs_in[key])[keep]
    if key in {"agentview_rgb", "eye_in_hand_rgb"}:
        data = _resize_rgb_batch(data)
    obs_out.create_dataset(key, data=data)


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.libero_raw_data_dir)
    target_dir = Path(args.libero_target_dir)

    if target_dir.exists() and any(target_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"target directory is not empty: {target_dir}; pass --overwrite")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    metainfo: dict[str, dict[str, dict[str, int | bool]]] = {}
    total_raw = 0
    total_kept = 0
    total_raw_steps = 0
    total_kept_steps = 0
    total_noops = 0

    for src_path in sorted(raw_dir.glob("*_demo.hdf5")):
        dst_path = target_dir / src_path.name
        task_name = src_path.name[: -len("_demo.hdf5")]
        metainfo[task_name] = {}
        with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
            src_data = src["data"]
            dst_data = dst.create_group("data")

            for demo_name in sorted(src_data.keys(), key=_demo_sort_key):
                demo = src_data[demo_name]
                actions = np.asarray(demo["actions"])
                keep = _filtered_indices(actions)
                total_raw += 1
                total_raw_steps += len(actions)
                total_noops += len(actions) - len(keep)

                if not keep:
                    metainfo[task_name][demo_name] = {
                        "kept": False,
                        "raw_steps": int(len(actions)),
                        "kept_steps": 0,
                        "filtered_noops": int(len(actions)),
                    }
                    continue

                out_demo = dst_data.create_group(demo_name)
                obs_out = out_demo.create_group("obs")
                obs_in = demo["obs"]
                for obs_key in obs_in.keys():
                    _copy_obs_dataset(obs_in, obs_out, obs_key, keep)

                kept_actions = actions[keep]
                out_demo.create_dataset("actions", data=kept_actions)
                for key in ["states", "robot_states"]:
                    if key in demo:
                        out_demo.create_dataset(key, data=np.asarray(demo[key])[keep])

                rewards = np.zeros(len(keep), dtype=np.uint8)
                dones = np.zeros(len(keep), dtype=np.uint8)
                rewards[-1] = 1
                dones[-1] = 1
                out_demo.create_dataset("rewards", data=rewards)
                out_demo.create_dataset("dones", data=dones)

                total_kept += 1
                total_kept_steps += len(keep)
                metainfo[task_name][demo_name] = {
                    "kept": True,
                    "raw_steps": int(len(actions)),
                    "kept_steps": int(len(keep)),
                    "filtered_noops": int(len(actions) - len(keep)),
                }

        print(f"{src_path.name}: wrote {dst_path}")

    summary = {
        "raw_demos": total_raw,
        "kept_demos": total_kept,
        "raw_steps": total_raw_steps,
        "kept_steps": total_kept_steps,
        "filtered_noops": total_noops,
        "image_resolution": IMAGE_RESOLUTION,
        "note": "keeps all demos with at least one non-noop action; no MuJoCo replay success filtering",
    }
    with open(target_dir / "metainfo.json", "w") as f:
        json.dump({"summary": summary, "tasks": metainfo}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
