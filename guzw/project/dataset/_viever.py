from __future__ import annotations

import argparse
from pathlib import Path
from random import choice, Random
import cv2
import matplotlib.pyplot as plt
import numpy as np


ACTION_NAMES = ["forward", "backward", "left", "right", "stop"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="View a random dataset sample.")
	parser.add_argument(
		"dataset_dir",
		nargs="?",
		default=Path(__file__).resolve().parent,
		type=Path,
		help="Directory containing sample_*.npz files.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional random seed for reproducible sample selection.",
	)
	parser.add_argument(
		"--sample",
		type=Path,
		default=None,
		help="View a specific .npz file instead of choosing one at random.",
	)
	return parser.parse_args()


def resolve_sample_path(dataset_dir: Path, sample_path: Path | None, seed: int | None) -> Path:
	if sample_path is not None:
		if sample_path.is_absolute():
			return sample_path
		return (dataset_dir / sample_path).resolve()

	sample_files = sorted(dataset_dir.glob("sample_*.npz"))
	if not sample_files:
		raise FileNotFoundError(f"No sample_*.npz files found in {dataset_dir}")

	if seed is None:
		return choice(sample_files)

	rng = Random(seed)
	return rng.choice(sample_files)


def load_sample(sample_path: Path) -> tuple[np.ndarray, int]:
	with np.load(sample_path) as data:
		states = data["states"]
		action_index = int(data["actions"])
	return states, action_index


def show_sample(sample_path: Path, states: np.ndarray, action_index: int) -> None:
	if states.ndim != 4:
		raise ValueError(f"Expected states to have 4 dimensions, got shape {states.shape}")

	frame_count = states.shape[0]
	columns = min(2, frame_count)
	rows = int(np.ceil(frame_count / columns))

	figure, axes = plt.subplots(rows, columns, figsize=(8 * columns, 5 * rows))
	axes_array = np.atleast_1d(axes).reshape(-1)

	for index, axis in enumerate(axes_array):
		if index < frame_count:
			frame = cv2.cvtColor(cv2.cvtColor(states[index], cv2.COLOR_RGB2HSV), cv2.COLOR_RGB2GRAY)
			axis.imshow(frame)
			axis.set_title(f"Frame {index + 1}")
			axis.axis("off")
		else:
			axis.axis("off")

	action_name = ACTION_NAMES[action_index] if 0 <= action_index < len(ACTION_NAMES) else f"unknown ({action_index})"
	figure.suptitle(f"{sample_path.name} | action: {action_name} ({action_index})", fontsize=14)
	figure.tight_layout()
	plt.show()


def main() -> None:
	args = parse_args()
	dataset_dir = args.dataset_dir.resolve()
	sample_path = resolve_sample_path(dataset_dir, args.sample, args.seed)
	states, action_index = load_sample(sample_path)
	print(f"Loaded {sample_path}")
	print(f"states shape: {states.shape}")
	print(f"action: {action_index} ({ACTION_NAMES[action_index] if action_index < len(ACTION_NAMES) else 'unknown'})")
	show_sample(sample_path, states, action_index)


if __name__ == "__main__":
	main()
