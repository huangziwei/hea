"""hea.translate — R ↔ hea source-to-source translator.

Pure-Python, no external R-parsing dependencies. See
``.claude/plans/r-translator.md`` for the full design.

The public surface (some pieces still pending — see the plan):

- :func:`translate_r` — R source → Python source (str → str).
- :func:`translate_py` — Python source → R source.
- :func:`from_R` — translate R source and execute it in the caller's frame
  (notebook entry point). Returns a result with ``.value`` and ``.source``.
- :func:`to_R` — translate Python source to R; optionally execute via the
  parity runner.
- :func:`parity` — run paired R + Python scripts and diff outputs; emits gaps
  to ``tests/parity/gaps.jsonl``.
"""
