import pathlib
import sys

SYNTHETIC_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYNTHETIC_DIR))
