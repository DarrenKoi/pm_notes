"""Ensure ``import opensearch_handler`` works for direct example runs."""

import sys
from pathlib import Path

_LIB_ROOT = str(Path(__file__).resolve().parent.parent.parent)  # Codes/python/
if _LIB_ROOT not in sys.path:
    sys.path.insert(0, _LIB_ROOT)
