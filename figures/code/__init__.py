"""Figure code for the paper.

Every module here writes its PDF into ``paper/figures/``, which is where the
manuscript expects them: ``paper/`` is the Overleaf project root, so a figure is
included as ``figures/name.pdf``. Data is read from ``results/`` and ``data/``
by relative path, so run from the repository root::

    python -m figures.code.paperfigs
    python -m figures.code.fig_classes

``sgl`` lives in ``src/`` and is put on the path here, so these modules can be
imported without installing the project.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
