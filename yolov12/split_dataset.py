#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / "dataset"
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a YOLO-style dataset into train/valid/test directories."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset root (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.2, 0.1),
        metavar=("TRAIN", "VALID", "TEST"),
        help="Split ratios for train, valid, and test",
    )
    return parser.parse_args()


def collect_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    image_lookup: dict[str, Path] = {}
    label_lookup: dict[str, Path] = {}

    for split in SPLITS:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue

        for path in images_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                if path.stem in image_lookup:
                    raise SystemExit(f"Duplicate image stem found: {path.stem}")
                image_lookup[path.stem] = path

        for path in labels_dir.iterdir():
            if path.is_file() and path.suffix == ".txt":
                if path.stem in label_lookup:
                    raise SystemExit(f"Duplicate label stem found: {path.stem}")
                label_lookup[path.stem] = path

    if not image_lookup and not label_lookup:
        flat_images_dir = dataset_dir / "images"
        flat_labels_dir = dataset_dir / "labels"
        if not flat_images_dir.is_dir() or not flat_labels_dir.is_dir():
            raise SystemExit(
                f"{dataset_dir} must contain either train/valid/test or flat images/ and labels/ directories"
            )

        for path in flat_images_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                image_lookup[path.stem] = path

        for path in flat_labels_dir.iterdir():
            if path.is_file() and path.suffix == ".txt":
                label_lookup[path.stem] = path

    image_stems = set(image_lookup)
    label_stems = set(label_lookup)
    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    if missing_labels or missing_images:
        problems = []
        if missing_labels:
            problems.append(f"missing labels for {len(missing_labels)} images")
        if missing_images:
            problems.append(f"missing images for {len(missing_images)} labels")
        raise SystemExit("Dataset pair mismatch: " + ", ".join(problems))

    stems = sorted(image_stems)
    if not stems:
        raise SystemExit("No image/label pairs found to split.")

    return [(image_lookup[stem], label_lookup[stem]) for stem in stems]


def prepare_output_dirs(dataset_dir: Path) -> None:
    for split in SPLITS:
        for kind in ("images", "labels"):
            split_dir = dataset_dir / split / kind
            split_dir.mkdir(parents=True, exist_ok=True)
            for leftover in split_dir.iterdir():
                if leftover.is_file():
                    leftover.unlink()
                elif leftover.is_dir():
                    shutil.rmtree(leftover)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()

    ratios = args.ratios
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise SystemExit("Ratios must be positive and sum to 1.0")

    pairs = collect_pairs(dataset_dir)
    prepare_output_dirs(dataset_dir)

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    total = len(pairs)
    train_end = round(total * ratios[0])
    valid_end = train_end + round(total * ratios[1])
    split_map = {
        "train": pairs[:train_end],
        "valid": pairs[train_end:valid_end],
        "test": pairs[valid_end:],
    }

    for split, split_pairs in split_map.items():
        for image_path, label_path in split_pairs:
            shutil.copy2(image_path, dataset_dir / split / "images" / image_path.name)
            shutil.copy2(label_path, dataset_dir / split / "labels" / label_path.name)

    print(
        f"Split {total} pairs with seed={args.seed}: "
        f"train={len(split_map['train'])}, valid={len(split_map['valid'])}, test={len(split_map['test'])}"
    )


if __name__ == "__main__":
    main()
