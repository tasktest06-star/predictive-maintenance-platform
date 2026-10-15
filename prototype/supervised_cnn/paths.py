"""Import bootstrap, so every script here can be run with ``python -I``.

``-I`` is mandatory for anything that reads the downloaded datasets: it runs in
isolated mode, which drops the script's own directory and the cwd from
``sys.path`` and ignores ``PYTHONPATH``. That is exactly the protection wanted
when a planted ``json.py`` in a download directory would otherwise be imported
in preference to the standard library.

The cost of ``-I`` is that ``PYTHONPATH=core:tools:edge`` stops working, so the
paths are reconstructed here from ``__file__`` instead -- the same trick
``slides/figures/make_core_figures.py`` and
``hardware/rpi5_hailo/tools/benchmark.py`` already use.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def bootstrap() -> str:
    """Put ``core``, ``edge``, ``prototype`` and ``production`` on the path."""
    for rel in ("core", "edge", "tools", "prototype", "production"):
        p = os.path.join(REPO, rel)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    return REPO


RESULTS_DIR = os.path.join(HERE, "results")
