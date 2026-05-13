"""No-op stand-ins for R base-graphics calls hea doesn't render yet.

The lmwr scripts pepper their plotting blocks with R-specific calls
(``pdf()`` / ``dev.off()`` to open and close PDF devices, ``image()``
for matrix heatmaps, ``stripchart()`` for 1-D scatter-by-group, ``gray()``
to build gray-scale palettes). Translating each to a real matplotlib
equivalent is a separate port; for now we accept the calls and do
nothing, so scripts can run end-to-end and surface their *real*
parity gaps instead of bouncing off graphics surface that's not the
point of the test.
"""

from __future__ import annotations


def pdf(file=None, height=None, width=None, **kwargs):
    """R: open a PDF graphics device. Stub — does not write a PDF.

    The translated script still runs every subsequent plotting call;
    you just won't get a PDF on disk. Pair with :func:`dev_off`.
    """
    return None


def dev_off(*args, **kwargs):
    """R: close the current graphics device. Stub — no-op."""
    return None


def image(*args, **kwargs):
    """R: ``image(matrix)`` — heatmap-style render of a matrix. Stub."""
    return None


def stripchart(*args, **kwargs):
    """R: 1-D scatter by group. Stub — no plot is drawn."""
    return None


def gray(level):
    """R: ``gray(level)`` — gray-scale color from level in [0, 1].

    Returns the hex string matplotlib accepts. ``gray(c(1, 0))`` is the
    R idiom for "white-to-black 2-stop palette" used with ``image()``.
    """
    if hasattr(level, "__iter__"):
        return [gray(v) for v in level]
    v = max(0.0, min(1.0, float(level)))
    byte = int(round(v * 255))
    return f"#{byte:02x}{byte:02x}{byte:02x}"
