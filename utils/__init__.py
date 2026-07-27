"""Utilities package initialization.

Provides backward-compatible module aliasing for legacy pickles that reference
``utils.dmdc_p`` while the implementation now lives in ``utils.pullback_dmdc``.
"""

import importlib
import sys


# Ensure legacy pickle module paths still resolve without keeping a dmdc_p.py file.
sys.modules.setdefault("utils.dmdc_p", importlib.import_module("utils.pullback_dmdc"))
