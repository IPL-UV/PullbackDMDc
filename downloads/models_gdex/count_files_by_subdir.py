#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys


def count_files(directory: Path, recursive: bool = True) -> int:
    if recursive:
        return sum(1 for path in directory.rglob("*") if path.is_file())
    return sum(1 for path in directory.iterdir() if path.is_file())


def subdirs_at_depth(root: Path, depth: int) -> list:
    current_level = [root]
    for _ in range(depth):
        next_level = []
        for directory in current_level:
            next_level.extend(path for path in directory.iterdir() if path.is_dir())
        current_level = next_level
    return sorted(current_level)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count files in subdirectories at a given depth from a root path."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="/data/databases/dmdc-variants/mmlea_v2",
        help="Root directory whose immediate subdirectories will be counted.",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Count only files directly inside each subdirectory (no recursion).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help=(
            "Subdirectory depth to report, relative to root. "
            "Default 2 (e.g., psl/cesm2)."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        print(f"Error: '{root}' is not a valid directory.", file=sys.stderr)
        return 1

    if args.depth < 1:
        print("Error: --depth must be >= 1.", file=sys.stderr)
        return 1

    subdirs = subdirs_at_depth(root, args.depth)
    if not subdirs:
        print(
            f"No subdirectories found at depth {args.depth} under '{root}'.",
            file=sys.stderr,
        )
        return 1

    for subdir in subdirs:
        file_count = count_files(subdir, recursive=not args.direct_only)
        key = subdir.relative_to(root).as_posix()
        print(f"{key}\t{file_count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
