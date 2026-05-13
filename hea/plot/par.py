"""Scoped ``par(mfrow=...)`` context manager — R's multi-panel layout idiom
without R's global statefulness.

R uses ``par(mfrow=c(nr, nc))`` to switch the device into a fill-the-grid
mode; subsequent plotting calls drop into the next cell. The state lives
on the device until the next ``par()`` call, so it leaks across script
boundaries. hea avoids that by scoping the grid to a ``with`` block::

    with par(mfrow=(1, 3)):
        hist(x, xlab="x")
        plot(density(x))
        plot(sort(x))

While inside the block, every base-graphics plotter called with
``ax=None`` pulls the next cell from the grid. Outside the block — or
when ``ax=`` is passed explicitly — behavior is unchanged.

Multi-panel plotters (``plot_lm``, ``pairs``, ``termplot``, multi-RHS
formulas) ignore ``par()`` and build their own figures: a 2×2 lm
diagnostic inside a ``par(mfrow=(1,3))`` would need nested gridspec,
which the simple stack here doesn't model.
"""

from __future__ import annotations

import numpy as np


_PAR_STACK: list = []


class _ParContext:
    """Allocator handed to plotters via the module-level ``_PAR_STACK``.

    On ``__enter__`` we build the figure + axes grid; each plotter that
    sees an active context calls :meth:`next_cell` to claim its axes.
    On ``__exit__`` we hide unused cells and run ``tight_layout`` for
    breathing room.
    """

    def __init__(
        self,
        *,
        mfrow: tuple[int, int] | None = None,
        mfcol: tuple[int, int] | None = None,
        figsize: tuple[float, float] | None = None,
    ):
        if mfrow is not None and mfcol is not None:
            raise ValueError("par(): pass mfrow= or mfcol=, not both.")
        if mfrow is None and mfcol is None:
            raise ValueError("par(): pass mfrow=(nrow,ncol) or mfcol=(nrow,ncol).")
        shape = mfrow if mfrow is not None else mfcol
        # Accept list too — R's ``par(mfrow=c(3,3))`` translates to a Python
        # list, and forcing the user to wrap in a tuple is needless friction.
        if isinstance(shape, list):
            shape = tuple(shape)
        if not (isinstance(shape, tuple) and len(shape) == 2
                and all(isinstance(x, int) and x > 0 for x in shape)):
            raise TypeError(
                f"par(): mfrow/mfcol must be a (nrow, ncol) tuple of "
                f"positive ints; got {shape!r}."
            )
        self.shape: tuple[int, int] = shape
        self.byrow: bool = mfrow is not None
        self.figsize = figsize
        self.fig = None
        self._cells: list = []
        self._idx: int = 0

    def __enter__(self) -> "_ParContext":
        import matplotlib.pyplot as plt

        nrow, ncol = self.shape
        figsize = self.figsize or (4.0 * ncol, 3.5 * nrow)
        self.fig, axarr = plt.subplots(nrow, ncol, figsize=figsize)
        # Flatten in fill order. plt.subplots returns a single Axes for
        # 1×1, a 1-D array for 1×N or N×1, and a 2-D array otherwise.
        arr = np.atleast_1d(np.asarray(axarr))
        if arr.ndim == 1:
            self._cells = list(arr)
        else:
            order = "C" if self.byrow else "F"
            self._cells = list(arr.ravel(order=order))
        _PAR_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _PAR_STACK.pop()
        # Hide cells that no plotter claimed — matches R's default of
        # leaving the trailing slots blank rather than stretching the
        # used ones.
        for ax in self._cells[self._idx:]:
            ax.set_visible(False)
        if self.fig is not None:
            try:
                self.fig.tight_layout()
            except Exception:
                # tight_layout occasionally fails on exotic legends /
                # colorbars; not worth raising over a layout polish.
                pass
        return False

    def next_cell(self):
        """Hand out the next axes in fill order, or raise if exhausted."""
        if self._idx >= len(self._cells):
            nrow, ncol = self.shape
            raise RuntimeError(
                f"par(mfrow={self.shape}): all {nrow * ncol} cells used. "
                f"Increase the grid or pass ax= explicitly for extra plots."
            )
        ax = self._cells[self._idx]
        self._idx += 1
        return ax

    @property
    def figure(self):
        return self.fig


def par(
    *,
    mfrow: tuple[int, int] | None = None,
    mfcol: tuple[int, int] | None = None,
    figsize: tuple[float, float] | None = None,
) -> _ParContext:
    """Open a scoped multi-panel grid. Use as ``with par(mfrow=(2, 3)):``.

    Parameters
    ----------
    mfrow
        ``(nrow, ncol)`` for row-major fill (R's ``par(mfrow=c(r, c))``).
    mfcol
        ``(nrow, ncol)`` for column-major fill (R's ``mfcol``). Pass
        exactly one of ``mfrow`` / ``mfcol``.
    figsize
        Optional matplotlib figure size in inches. Defaults to
        ``(4*ncol, 3.5*nrow)``.

    Inside the ``with`` block, every base-graphics plotter called with
    ``ax=None`` consumes the next cell of the grid. Passing ``ax=``
    explicitly bypasses the grid. Multi-panel plotters (``plot_lm``,
    ``pairs``, ``termplot``) always build their own figures.
    """
    return _ParContext(mfrow=mfrow, mfcol=mfcol, figsize=figsize)


def _current_par() -> _ParContext | None:
    """Return the innermost active ``par()`` context, or ``None``."""
    return _PAR_STACK[-1] if _PAR_STACK else None
