#!/usr/bin/env python3
"""Convert regenerated LIBERO no-noop HDF5 demonstrations to TFDS/RLDS."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from libero.libero import benchmark


try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


IMAGE_SHAPE = (256, 256, 3)
STATE_DIM = 8
ACTION_DIM = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero_task_suite",
        required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
    )
    parser.add_argument(
        "--hdf5_dir",
        required=True,
        help="Directory produced by regenerate_libero_no_noops.py, containing *_demo.hdf5 files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Final TFDS builder directory, for example data/libero/libero_spatial_no_noops/1.0.0.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_rotate_images",
        action="store_true",
        help="Keep raw HDF5 image orientation. By default images are rotated 180 degrees.",
    )
    return parser.parse_args()


def _demo_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else -1, name)


def _suite_tasks(suite: str) -> dict[str, str]:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    return {task_suite.get_task(i).name: task_suite.get_task(i).language for i in range(task_suite.n_tasks)}


def _episode_state(obs_group: h5py.Group, idx: int) -> np.ndarray:
    ee_state = np.asarray(obs_group["ee_states"][idx], dtype=np.float32).reshape(-1)
    gripper_state = np.asarray(obs_group["gripper_states"][idx], dtype=np.float32).reshape(-1)
    state = np.concatenate([ee_state, gripper_state], axis=0)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"expected state shape {(STATE_DIM,)}, got {state.shape}")
    return state


def _image(arr: np.ndarray, rotate: bool) -> np.ndarray:
    img = np.asarray(arr)
    if rotate:
        img = np.rot90(img, 2)
    if img.shape != IMAGE_SHAPE:
        raise ValueError(f"expected image shape {IMAGE_SHAPE}, got {img.shape}")
    return img.astype(np.uint8, copy=False)


class LiberoRlds(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Regenerated LIBERO no-noop demonstrations in RLDS-style format."}

    def __init__(self, *, hdf5_dir: str, suite: str, rotate_images: bool, **kwargs):
        self.hdf5_dir = Path(hdf5_dir)
        self.suite = suite
        self.rotate_images = bool(rotate_images)
        self.task_to_language = _suite_tasks(suite)
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "episode_id": tfds.features.Text(),
                    "task_name": tfds.features.Text(),
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(shape=IMAGE_SHAPE, dtype=np.uint8),
                                    "wrist_image": tfds.features.Image(shape=IMAGE_SHAPE, dtype=np.uint8),
                                    "state": tfds.features.Tensor(shape=(STATE_DIM,), dtype=np.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(ACTION_DIM,), dtype=np.float32),
                            "reward": np.float32,
                            "discount": np.float32,
                            "is_first": np.bool_,
                            "is_last": np.bool_,
                            "is_terminal": np.bool_,
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager):
        del dl_manager
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for hdf5_path in sorted(self.hdf5_dir.glob("*_demo.hdf5")):
            task_name = hdf5_path.name[: -len("_demo.hdf5")]
            language = self.task_to_language.get(task_name, task_name.replace("_", " "))
            with h5py.File(hdf5_path, "r") as f:
                data = f["data"]
                for demo_name in sorted(data.keys(), key=_demo_sort_key):
                    demo = data[demo_name]
                    obs = demo["obs"]
                    actions = np.asarray(demo["actions"], dtype=np.float32)
                    rewards = np.asarray(demo.get("rewards", np.zeros(len(actions))), dtype=np.float32)
                    dones = np.asarray(demo.get("dones", np.zeros(len(actions))), dtype=bool)
                    steps = []
                    for i, action in enumerate(actions):
                        steps.append(
                            {
                                "observation": {
                                    "image": _image(obs["agentview_rgb"][i], self.rotate_images),
                                    "wrist_image": _image(obs["eye_in_hand_rgb"][i], self.rotate_images),
                                    "state": _episode_state(obs, i),
                                },
                                "action": action.reshape(ACTION_DIM),
                                "reward": np.float32(rewards[i] if i < len(rewards) else 0.0),
                                "discount": np.float32(0.0 if i == len(actions) - 1 else 1.0),
                                "is_first": i == 0,
                                "is_last": i == len(actions) - 1,
                                "is_terminal": bool(dones[i]) if i < len(dones) else i == len(actions) - 1,
                                "language_instruction": language,
                            }
                        )
                    if steps:
                        episode_id = f"{task_name}/{demo_name}"
                        yield episode_id, {"episode_id": episode_id, "task_name": task_name, "steps": steps}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output_dir is not empty: {output_dir}; pass --overwrite to replace it")

    if output_dir.name != "1.0.0":
        raise ValueError("output_dir should be the final TFDS version directory ending in 1.0.0")

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="libero_rlds_build_") as tmp:
        builder = LiberoRlds(
            hdf5_dir=args.hdf5_dir,
            suite=args.libero_task_suite,
            rotate_images=not args.no_rotate_images,
            data_dir=tmp,
        )
        builder.download_and_prepare()
        built_dir = Path(builder.data_dir)

        tmp_output = output_parent / f".{output_dir.name}.tmp"
        if tmp_output.exists():
            shutil.rmtree(tmp_output)
        shutil.copytree(built_dir, tmp_output)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(tmp_output, output_dir)

    print(f"Wrote TFDS/RLDS builder directory: {output_dir}")


if __name__ == "__main__":
    main()
