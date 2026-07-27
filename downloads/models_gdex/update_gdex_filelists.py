#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


FILELIST_START_RE = re.compile(r"^\s*filelist\s*=\s*\[", re.IGNORECASE)


def transform_lines(lines):
    output = []
    in_filelist = False
    bracket_depth = 0
    removed_smbb = 0
    commented_non_historical = 0

    for line in lines:
        if not in_filelist:
            output.append(line)
            if FILELIST_START_RE.match(line):
                in_filelist = True
                bracket_depth = line.count("[") - line.count("]")
            continue

        bracket_depth += line.count("[") - line.count("]")

        if "smbb" in line.lower():
            removed_smbb += 1
        else:
            stripped = line.strip()
            is_closing_bracket = stripped.startswith("]")
            should_comment = (
                stripped
                and not stripped.startswith("#")
                and not is_closing_bracket
                and "historical" not in line.lower()
            )

            if should_comment:
                leading_ws = line[: len(line) - len(line.lstrip())]
                line = f"{leading_ws}# {line.lstrip()}"
                commented_non_historical += 1

            output.append(line)

        if bracket_depth <= 0:
            in_filelist = False

    return output, removed_smbb, commented_non_historical


def process_file(path):
    original = path.read_text(encoding="utf-8")
    new_lines, removed_smbb, commented_non_historical = transform_lines(
        original.splitlines(keepends=True)
    )
    updated = "".join(new_lines)
    changed = updated != original

    if changed:
        path.write_text(updated, encoding="utf-8")

    return changed, removed_smbb, commented_non_historical


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update gdex download scripts: remove lines containing 'smbb' and "
            "comment out non-'historical' lines in filelist."
        )
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory containing gdex-download (*.py) files (default: current directory).",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir).expanduser().resolve()
    targets = sorted(base_dir.glob("gdex-download (*.py"))

    if not targets:
        print(f"No matching files found in {base_dir}")
        return

    changed_count = 0
    total_removed = 0
    total_commented = 0

    for path in targets:
        changed, removed_smbb, commented_non_historical = process_file(path)
        status = "updated" if changed else "no changes"
        print(
            f"{path.name}: {status} | removed smbb: {removed_smbb}, "
            f"commented non-historical: {commented_non_historical}"
        )
        if changed:
            changed_count += 1
        total_removed += removed_smbb
        total_commented += commented_non_historical

    print(
        f"Done. Files changed: {changed_count}/{len(targets)} | "
        f"total smbb removed: {total_removed} | total non-historical commented: {total_commented}"
    )


if __name__ == "__main__":
    main()