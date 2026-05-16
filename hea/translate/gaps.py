"""Gap registry — append-only JSONL of translator / parity failures.

Each row records one observed divergence between an R script's output and
the hea translation's output, OR one translation failure (parse error,
unknown function, etc). The registry doubles as a parity backlog:
closing a gap means fixing hea/the translator/the registry until the row
can be marked ``status: closed`` (or deleted).

Schema (one JSON object per line):

```json
{
  "id":          "string — derived from (kind, subject, source); stable across runs",
  "kind":        "string — see _KNOWN_KINDS below",
  "subject":     "string — what the gap is about (function name, column name, …)",
  "source":      "string — file path + line/byte range when applicable",
  "snippet":     "string — relevant R source slice (≤200 chars)",
  "translation": "string — relevant Python source slice (≤200 chars)",
  "status":      "open | closed | wontfix",
  "first_seen":  "ISO date — when this gap first appeared",
  "last_seen":   "ISO date — when this gap was most recently observed",
  "notes":       "string — free-text (root cause, workaround, etc)"
}
```

Storage: ``tests/parity/gaps.jsonl`` (relative to repo root). The file is
created on first ``log_gap`` if absent. Each call appends; deduplication
keys on ``(kind, subject, source)`` so re-running the same parity check
does not balloon the file.

Why JSONL not SQLite: humans need to read, grep, and hand-edit gaps as
parity work proceeds.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


_KNOWN_KINDS: frozenset[str] = frozenset({
    # Translation-side
    "parse_error",
    "unknown_function",
    "unknown_verb",
    "unknown_arg",
    "unknown_geom",
    "unknown_scale",
    "unknown_coord",
    "unknown_facet",
    "unknown_theme_element",
    "nse_ambiguous",
    "not_implemented",
    "replacement_function",   # R `f(x) <- v` setter form
    "with_expression",        # `with(df, expr)` — NSE rewrite not yet built
    "python_keyword_call",    # bare `class(x)`/`lambda(x)` collides with a Python keyword
    "lexer_ambiguity",        # parser/lexer can't disambiguate the input
    # Runtime-side
    "runtime_error_r",
    "runtime_error_py",
    # Parity-side
    "result_diff_schema",
    "result_diff_values",
    "result_diff_factor",
    "result_diff_plot_data",
    "result_diff_row_count",
})


_DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent.parent / "tests" / "parity" / "gaps.jsonl"


@dataclass(slots=True)
class Gap:
    """One row in the registry. ``id`` is auto-derived from the dedup key
    if left blank."""

    kind: str
    subject: str
    source: str = ""
    snippet: str = ""
    translation: str = ""
    status: str = "open"
    first_seen: str = ""
    last_seen: str = ""
    notes: str = ""
    id: str = ""

    def __post_init__(self):
        if self.kind not in _KNOWN_KINDS:
            raise ValueError(
                f"unknown gap kind {self.kind!r}; pick from {sorted(_KNOWN_KINDS)}"
            )
        if not self.id:
            self.id = _make_id(self.kind, self.subject, self.source)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Gap":
        return cls(**d)


def _make_id(kind: str, subject: str, source: str) -> str:
    """Stable hash-based id over the dedup key. 12 hex chars is enough to
    avoid collisions in a registry of even tens of thousands of rows.
    """
    h = hashlib.sha256(f"{kind}\x00{subject}\x00{source}".encode("utf-8")).hexdigest()
    return h[:12]


def _today() -> str:
    return _dt.date.today().isoformat()


def log_gap(
    *,
    kind: str,
    subject: str,
    source: str = "",
    snippet: str = "",
    translation: str = "",
    notes: str = "",
    registry: Optional[Path] = None,
) -> Gap:
    """Record a gap. Returns the resulting :class:`Gap` (after dedup merge).

    Behavior:
    - If a row with the same ``id`` already exists, only ``last_seen`` is
      updated (and ``notes`` is appended if non-empty and new). The
      existing ``status`` is preserved so closed gaps stay closed when
      observed again.
    - Otherwise, a new row is appended with ``first_seen = last_seen = today``
      and ``status = "open"``.
    """
    registry = registry or _DEFAULT_REGISTRY
    today = _today()
    new = Gap(
        kind=kind,
        subject=subject,
        source=source,
        snippet=snippet,
        translation=translation,
        notes=notes,
        first_seen=today,
        last_seen=today,
    )
    rows = read_gaps(registry)
    existing = next((g for g in rows if g.id == new.id), None)
    if existing is None:
        rows.append(new)
        _write_all(rows, registry)
        return new
    # Merge into existing
    existing.last_seen = today
    if notes and notes not in existing.notes:
        existing.notes = (existing.notes + "\n" + notes).strip() if existing.notes else notes
    _write_all(rows, registry)
    return existing


def read_gaps(registry: Optional[Path] = None) -> list[Gap]:
    """Load all rows from ``registry``. Returns an empty list if missing."""
    registry = registry or _DEFAULT_REGISTRY
    if not registry.exists():
        return []
    rows: list[Gap] = []
    for line in registry.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(Gap.from_dict(json.loads(line)))
    return rows


def find_gap(gap_id: str, registry: Optional[Path] = None) -> Optional[Gap]:
    """Look up a row by id. Returns ``None`` if absent."""
    for g in read_gaps(registry):
        if g.id == gap_id:
            return g
    return None


def _write_all(rows: list[Gap], registry: Path) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(g.to_dict(), separators=(",", ":")) for g in rows]
    registry.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
