"""Run the synthetic test suite without pytest.

    python run_tests.py                 # every test
    python run_tests.py --only forcing  # only tests whose module or name contains "forcing"
    python run_tests.py --list          # list what would run

Imports every tests/test_*.py and calls its module-level test_* functions in definition order.
The tests are plain functions with asserts, so no test framework is needed. A failure is reported
with its traceback and the run continues; the exit status is 1 if anything failed.

tests/test_synthetic_pipeline.py is skipped here: it defines no test_* functions. It is a script,
run it directly (run_synthetic_tests.sh does).
"""

import argparse
import importlib.util
import pathlib
import sys
import time
import traceback

SYNTHETIC_DIR = pathlib.Path(__file__).resolve().parent
TESTS_DIR = SYNTHETIC_DIR / "tests"
SKIP = {"test_synthetic_pipeline"}

# Same two entries tests/conftest.py and the test modules rely on: the package dir for
# `from ablation_data import ...` and the tests dir for `from support import ...`.
for path in (TESTS_DIR, SYNTHETIC_DIR):
    sys.path.insert(0, str(path))


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def collect(only=None):
    """(module_name, test_name, function) for every test, modules and tests in definition order."""
    found = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        if path.stem in SKIP:
            continue
        module = load(path)
        for name, fn in vars(module).items():  # insertion order == definition order
            if name.startswith("test_") and callable(fn) and getattr(fn, "__module__", None) == path.stem:
                if only is None or only in name or only in path.stem:
                    found.append((path.stem, name, fn))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="substring filter on the module or test name")
    parser.add_argument("--list", action="store_true", help="list the tests and exit")
    args = parser.parse_args()

    tests = collect(args.only)
    if not tests:
        sys.exit(f"no tests matched {args.only!r}")
    if args.list:
        for module, name, _ in tests:
            print(f"{module}::{name}")
        return

    failures, current = [], None
    start = time.time()
    for module, name, fn in tests:
        if module != current:
            current = module
            print(f"\n{module}")
        t = time.time()
        try:
            fn()
        except Exception:
            failures.append((module, name, traceback.format_exc()))
            print(f"  FAIL  {name} ({time.time() - t:.1f}s)")
        else:
            print(f"  ok    {name} ({time.time() - t:.1f}s)")

    for module, name, tb in failures:
        print(f"\n{'=' * 70}\nFAIL {module}::{name}\n{'=' * 70}\n{tb}", end="")
    print(f"\n{len(tests) - len(failures)} passed, {len(failures)} failed in {time.time() - start:.1f}s")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
