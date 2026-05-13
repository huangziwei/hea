"""R's named numeric vector — the return type for ``coef()``, ``apply()``,
and any other R surface whose result is a values-with-names sequence.

R's named vector supports a unique mix of operations:

- Positional indexing, 1-based: ``v[1]`` is the first element.
- Name indexing: ``v["x"]`` looks up by name.
- Slicing: ``v[1:5]`` returns positions 1..5 (inclusive both ends — R's
  ``i:j`` is an explicit integer sequence, which the translator emits
  as ``seq(i, j)`` = numpy array of length j-i+1, so the slice here
  goes through fancy-indexing).
- Elementwise arithmetic with scalar, list, ndarray, or another
  ``NamedVector``. Names of the LHS are preserved; R's recycling rule
  applies when lengths differ.
- ``len(v)``, iteration over values, ``list(v)`` → values.

The translator returns ``NamedVector`` from ``coef()`` and from ``c()``
when any input is named (a plain numeric vector / numpy array is
returned otherwise). Downstream Python users can use it as a list of
values via ``.values`` (numpy array) or as a dict via ``dict(zip(v.names, v.values))``.
"""

from __future__ import annotations

import numpy as np


class NamedVector:
    """R-style named numeric vector. Indexing is 1-based on integers,
    name-based on strings."""

    __slots__ = ("names", "values")

    def __init__(self, names, values):
        self.names = list(names)
        self.values = np.asarray(values, dtype=float).ravel()
        if len(self.names) != len(self.values):
            raise ValueError(
                f"NamedVector: {len(self.names)} names vs "
                f"{len(self.values)} values"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "NamedVector":
        """Build from a name→value mapping (preserves insertion order)."""
        return cls(list(d.keys()), list(d.values()))

    # ---- container protocol ----------------------------------------

    def __len__(self) -> int:
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __contains__(self, item) -> bool:
        return item in self.names

    def __getitem__(self, key):
        # Name lookup → scalar.
        if isinstance(key, str):
            i = self.names.index(key)
            return float(self.values[i])

        # 0-based integer scalar → length-1 NamedVector. (hea is 0-based
        # throughout — the translator's job is to shift R's 1-based
        # indices when emitting Python, not the runtime container's.)
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            i = int(key)
            return NamedVector([self.names[i]], [self.values[i]])

        # Python slice → 0-based exclusive (Python convention).
        if isinstance(key, slice):
            return NamedVector(self.names[key], self.values[key])

        # Iterable of int / bool / str → fancy index, 0-based.
        if isinstance(key, (list, tuple, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == bool:
                idx = np.where(arr)[0]
                return NamedVector(
                    [self.names[int(i)] for i in idx],
                    self.values[idx],
                )
            if arr.dtype.kind in ("i", "u"):
                idx = [int(i) for i in arr]
                return NamedVector(
                    [self.names[i] for i in idx],
                    self.values[idx],
                )
            if arr.dtype.kind in ("U", "O"):
                idx = [self.names.index(str(n)) for n in arr]
                return NamedVector(
                    [self.names[i] for i in idx],
                    self.values[idx],
                )

        raise TypeError(
            f"NamedVector: invalid index type {type(key).__name__}"
        )

    # ---- arithmetic ------------------------------------------------

    def _binop(self, other, op):
        a = self.values
        if isinstance(other, NamedVector):
            b = other.values
            n = max(len(a), len(b))
            a_, b_ = np.resize(a, n), np.resize(b, n)
            names = self.names if len(self) >= len(other) else other.names
            if len(names) < n:
                names = (names * ((n // len(names)) + 1))[:n]
            return NamedVector(names, op(a_, b_))
        b = np.asarray(other)
        return NamedVector(self.names, op(a, b))

    def __add__(self, other):
        return self._binop(other, np.add)

    def __radd__(self, other):
        return self._binop(other, lambda x, y: np.add(y, x))

    def __sub__(self, other):
        return self._binop(other, np.subtract)

    def __rsub__(self, other):
        return self._binop(other, lambda x, y: np.subtract(y, x))

    def __mul__(self, other):
        return self._binop(other, np.multiply)

    def __rmul__(self, other):
        return self._binop(other, lambda x, y: np.multiply(y, x))

    def __truediv__(self, other):
        return self._binop(other, np.true_divide)

    def __rtruediv__(self, other):
        return self._binop(other, lambda x, y: np.true_divide(y, x))

    def __neg__(self):
        return NamedVector(self.names, -self.values)

    def __pos__(self):
        return NamedVector(self.names, +self.values)

    # ---- representation -------------------------------------------

    def __repr__(self) -> str:
        if not self.names:
            return "NamedVector()"
        # R-style two-line print: names row, values row, column-aligned.
        vals = [_format_num(v) for v in self.values]
        widths = [max(len(n), len(v)) for n, v in zip(self.names, vals)]
        name_row = " ".join(n.rjust(w) for n, w in zip(self.names, widths))
        val_row = " ".join(v.rjust(w) for v, w in zip(vals, widths))
        return f"{name_row}\n{val_row}"

    def to_dict(self) -> dict:
        """Convert to a plain ``{name: value}`` dict."""
        return dict(zip(self.names, self.values.tolist()))

    # ---- numpy / polars interop -----------------------------------

    def __array__(self, dtype=None):
        """Allow ``np.asarray(nv)`` — return the values."""
        if dtype is None:
            return self.values
        return self.values.astype(dtype)


def _format_num(v: float, digits: int = 6) -> str:
    """Compact numeric format matching R's default `signif`-like output."""
    if v != v:  # NaN
        return "NaN"
    if v == 0:
        return "0"
    s = f"{v:.{digits}g}"
    return s
