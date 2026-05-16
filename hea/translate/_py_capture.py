"""Python-side capture driver for the parity runner.

Run as a subprocess:

    python -m hea.translate._py_capture <user_script.py> <out.csv> <out.schema.json>

Behaviour:

1. Parse ``user_script.py`` with stdlib ``ast``.
2. If the last top-level statement is an ``Expr``, rewrite it as
   ``__hea_last__ = <expr>`` so its value is recoverable.
3. Build a namespace with every public name from ``hea`` pre-imported
   (so the translated script doesn't need its own ``from hea import ...``
   preamble — same convention as ``hea.from_R`` will use in Phase 9).
4. ``exec`` the rewritten AST.
5. Read ``__hea_last__`` from the namespace. If it's a hea/polars
   DataFrame, serialize to CSV + a schema JSON sidecar capturing
   dtypes and factor levels. Otherwise: wrap the scalar / vector in a
   one-column DataFrame and serialize the same way.

Exit code: 0 on success, 1 on any failure (with traceback on stderr).
"""

from __future__ import annotations

import ast
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _rewrite_last_expr(tree: ast.Module) -> None:
    """Mutate ``tree`` so the last top-level Expr binds to ``__hea_last__``.
    If there's no trailing Expr, append ``__hea_last__ = None`` instead."""
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        new = ast.Assign(
            targets=[ast.Name(id="__hea_last__", ctx=ast.Store())],
            value=last.value,
        )
        ast.copy_location(new, last)
        tree.body[-1] = new
    else:
        tree.body.append(
            ast.Assign(
                targets=[ast.Name(id="__hea_last__", ctx=ast.Store())],
                value=ast.Constant(None),
            )
        )
    ast.fix_missing_locations(tree)


def _build_namespace() -> dict:
    """Caller's globals — every public ``hea`` name (top-level + every
    user-facing sub-namespace) pre-bound so the user script can write
    ``col("x")`` / ``case_when(...)`` / ``lm(...)`` etc. with no imports."""
    import hea  # local import — keeps cli startup fast

    ns: dict[str, Any] = {"__name__": "__capture__", "hea": hea}
    for name in dir(hea):
        if not name.startswith("_"):
            ns[name] = getattr(hea, name)
    # The top-level only exposes polars + the three subclasses; everything
    # else (verbs, models, families, R functions, …) lives under a
    # sub-namespace. Pre-bind those too.
    for sub in (hea.tidy, hea.models, hea.family, hea.R,
                hea.data, hea.session_info):
        for name in dir(sub):
            if not name.startswith("_") and name not in ns:
                ns[name] = getattr(sub, name)
    return ns


def _serialize_result(result: Any, out_csv: Path, out_schema: Path) -> None:
    """Write ``result`` to CSV + schema JSON, both readable by R."""
    import polars as pl

    if isinstance(result, pl.DataFrame):
        df = result
    elif isinstance(result, pl.Series):
        df = result.to_frame()
    elif result is None:
        out_csv.write_text("")
        out_schema.write_text(json.dumps({"dtypes": {}, "factors": {}}))
        return
    else:
        # Scalar / list / dict — wrap as a one-row, one-column frame
        # so the diff path stays homogeneous.
        try:
            df = pl.DataFrame({"value": [result]})
        except Exception as e:
            raise RuntimeError(
                f"cannot serialize result of type {type(result).__name__}: {e}"
            ) from e

    df.write_csv(out_csv)

    schema = {
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "factors": _capture_factors(df),
        "shape": list(df.shape),
    }
    out_schema.write_text(json.dumps(schema, indent=2))


def _capture_factors(df) -> dict:
    """Extract Enum/Categorical column levels so the diff can compare them."""
    import polars as pl

    out: dict = {}
    for col, dtype in zip(df.columns, df.dtypes):
        if isinstance(dtype, pl.Enum):
            levels = list(dtype.categories)
            out[col] = {"levels": levels, "ordered": False}
        elif isinstance(dtype, pl.Categorical):
            levels = list(df[col].cat.get_categories())
            out[col] = {"levels": levels, "ordered": False}
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            f"usage: {argv[0]} <user_script.py> <out.csv> <out.schema.json>",
            file=sys.stderr,
        )
        return 2

    user_script = Path(argv[1])
    out_csv = Path(argv[2])
    out_schema = Path(argv[3])

    try:
        src = user_script.read_text(encoding="utf-8")
        tree = ast.parse(src, str(user_script))
        _rewrite_last_expr(tree)
        ns = _build_namespace()
        exec(compile(tree, str(user_script), "exec"), ns)
        result = ns.get("__hea_last__")
        _serialize_result(result, out_csv, out_schema)
    except Exception:
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
