"""hea.translate — R ↔ hea source-to-source translator.

Pure-Python, no external R-parsing dependencies.

Public surface:

* :func:`from_R` — translate R source to Python; optionally execute it
  in the caller's frame. Returns a :class:`Result` with ``.value``,
  ``.source``, and ``.gaps``.
* :func:`to_R` — translate Python source to R; optionally execute via
  the parity runner. Returns the same :class:`Result` shape.
* :func:`translate_r` — pure R source → Python source (string only).
* :func:`translate_py` — pure Python source → R source (string only).
* :func:`parity` — run paired R + Python scripts and diff outputs;
  emits gaps to ``tests/parity/gaps.jsonl``.
* :class:`Result` — the return type of :func:`from_R` / :func:`to_R`.
* :class:`HeaTranslationGap` — warning category for translator gaps.
"""

from .inline import HeaTranslationGap, Result, from_R, to_R
from .py_to_r import translate as translate_py
from .r_to_py import translate as translate_r
from .runner import parity

__all__ = [
    "from_R", "to_R",
    "translate_r", "translate_py",
    "parity",
    "Result", "HeaTranslationGap",
]
