"""Wilkinson-style extended axis labelling — port of R's ``labeling::extended``.

Reference: Talbot, Lin & Hanrahan, *An Extended Algorithm for Axis Labeling*,
IEEE Vis 2010. ggplot2's ``scale_x/y_continuous`` defaults route through
this algorithm. Our port mirrors the R sources line-for-line so that
oracle parity is achievable when we wire R-oracle dumps later.

The score combines four components:

* **simplicity** — preference for round step sizes (1, 5, 2, 2.5, 4, 3) and
  inclusion of zero;
* **coverage** — how well the break range covers the data range;
* **density** — how close the number of breaks is to the target ``m``;
* **legibility** — placeholder constant 1 (full legibility scoring is
  layout-dependent in the paper; our axes inherit matplotlib's font metrics).
"""

from __future__ import annotations

import numpy as np


def extended_breaks(
    dmin: float,
    dmax: float,
    m: int = 5,
    Q: tuple = (1, 5, 2, 2.5, 4, 3),
    only_loose: bool = False,
    weights: tuple = (0.25, 0.2, 0.5, 0.05),
) -> np.ndarray:
    """Compute "nice" tick positions covering ``[dmin, dmax]``.

    ``m`` is the target tick count; ``Q`` is the preferred step granularities
    (most-preferred first). ``only_loose=True`` requires breaks to bracket
    the data; otherwise tight breaks (inside the data range) are allowed.
    """
    if dmin > dmax:
        dmin, dmax = dmax, dmin
    if dmax - dmin < 1e-15:
        return np.array([dmin])

    n_Q = len(Q)
    w = weights

    best_lmin = None
    best_lmax = None
    best_lstep = None
    best_score = -2.0

    j = 1
    while j < 100:
        skip_j = False
        for qi, q in enumerate(Q):
            sm = _simplicity_max(qi, n_Q, j)
            if (w[0] * sm + w[1] + w[2] + w[3]) < best_score:
                skip_j = True
                break

            k = 2
            while k < 100:
                dm = _density_max(k, m)
                if (w[0] * sm + w[1] + w[2] * dm + w[3]) < best_score:
                    break

                delta = (dmax - dmin) / (k + 1) / j / q
                z = int(np.ceil(np.log10(delta))) if delta > 0 else -300

                while z < 300:
                    step = j * q * 10.0 ** z
                    cm = _coverage_max(dmin, dmax, step * (k - 1))
                    if (w[0] * sm + w[1] * cm + w[2] * dm + w[3]) < best_score:
                        break

                    min_start = int(np.floor(dmax / step) * j - (k - 1) * j)
                    max_start = int(np.ceil(dmin / step) * j)

                    if min_start > max_start:
                        z += 1
                        continue

                    for start in range(min_start, max_start + 1):
                        lmin = start * (step / j)
                        lmax = lmin + step * (k - 1)
                        lstep = step

                        s = _simplicity(qi, n_Q, j, lmin, lmax, lstep)
                        c = _coverage(dmin, dmax, lmin, lmax)
                        g = _density(k, m, dmin, dmax, lmin, lmax)
                        leg = 1.0

                        score = w[0] * s + w[1] * c + w[2] * g + w[3] * leg

                        if score > best_score and (
                            not only_loose or (lmin <= dmin and lmax >= dmax)
                        ):
                            best_lmin = lmin
                            best_lmax = lmax
                            best_lstep = lstep
                            best_score = score
                    z += 1
                k += 1
        if skip_j:
            break
        j += 1

    if best_lmin is None:
        return np.array([dmin, dmax])

    n = int(round((best_lmax - best_lmin) / best_lstep)) + 1
    return best_lmin + best_lstep * np.arange(n)


# ---------------------------------------------------------------------------
# scoring components — direct ports of labeling::extended internals
# ---------------------------------------------------------------------------

def _simplicity(qi: int, n_Q: int, j: int, lmin: float, lmax: float, lstep: float) -> float:
    eps = 1e-10
    # +1 if zero is inside the labeled range and the break grid passes through it
    v = 1.0 if (abs(lmin - lstep * np.round(lmin / lstep)) < eps
                and lmin <= 0 <= lmax) else 0.0
    return 1.0 - qi / (n_Q - 1) - j + v


def _simplicity_max(qi: int, n_Q: int, j: int) -> float:
    return 1.0 - qi / (n_Q - 1) - j + 1.0


def _coverage(dmin: float, dmax: float, lmin: float, lmax: float) -> float:
    range_ = dmax - dmin
    return 1.0 - 0.5 * (
        (dmax - lmax) ** 2 + (dmin - lmin) ** 2
    ) / (0.1 * range_) ** 2


def _coverage_max(dmin: float, dmax: float, span: float) -> float:
    range_ = dmax - dmin
    if span > range_:
        half = (span - range_) / 2.0
        return 1.0 - 0.5 * (half ** 2 + half ** 2) / (0.1 * range_) ** 2
    return 1.0


def _density(k: int, m: int, dmin: float, dmax: float, lmin: float, lmax: float) -> float:
    r = (k - 1) / (lmax - lmin)
    rt = (m - 1) / (max(lmax, dmax) - min(dmin, lmin))
    return 2.0 - max(r / rt, rt / r)


def _density_max(k: int, m: int) -> float:
    if k >= m:
        return 2.0 - (k - 1) / (m - 1)
    return 1.0
