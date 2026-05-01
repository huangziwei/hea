"""Discrete-covariate machinery for ``bam(..., discrete=TRUE)``.

Mirrors the discrete pipeline in ``mgcv/R/bam.r``:

* :func:`compress_df` — discretise one variable (``compress.df``,
  bam.r:122-184). Numeric variables are rounded to a regular grid; factor
  levels deduplicated; matrix arguments (mgcv summation convention) are
  pooled across columns. Returns the unique discretised values plus an
  integer index ``k`` mapping every original observation back to a row of
  the unique table (``k`` is a matrix when the input is a matrix).
* :func:`check_term` — bookkeeping helper for :func:`discrete_mf`
  (bam.r:185-200).
* :func:`discrete_mf` — walk a parsed formula's smooth specs and
  discretise each marginal variable, building the index table ``k``,
  the per-variable column ranges ``ks``, and the discretised model frame
  ``mf`` (bam.r:201-380).

Plus the design-side machinery:

* :func:`build_discrete_design` — given an already-fit set of
  :class:`~hea.formula.SmoothBlock` (carrying basis specs from a
  representative subsample fit) and a discretised model frame, build the
  per-marginal design blocks ``Xd``, the term/marginal indexing tables
  (``ts``, ``dt``), the constraint vectors ``v``, and the per-term
  coef-slice mapping ``lpip``. Mirrors the C-side packed layout consumed
  by ``XWXd``/``XWyd``/``Xbd`` (mgcv ``misc.r``).
* :func:`Xbd`, :func:`XWXd`, :func:`XWyd` — vectorised numpy ports of
  the discrete C kernels (mgcv ``src/discrete.c``). Output is byte-equal
  (modulo floating-point) to the full-design oracle on small examples;
  see ``tests/test_discrete_kernels.py``.

The kernels are the *substance* of the discrete fitter — they replace
the chunked-QR build in :func:`hea.bam._build_qr_chunked_pirls` with
direct ``X'WX``/``X'Wy``/``Xβ`` formation on the compressed
representation, slashing per-PIRLS-step memory and FLOPs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import polars as pl

from ._r_random import RMersenneTwister
from .design import is_matrix_col, matrix_to_2d
from .formula import (
    BasisSpec,
    SmoothBlock,
    _RawBasis,
    _LinearTransformRawBasis,
    _TensorRawBasis,
    _T2RawBasis,
    _T2PredictRawBasis,
)


__all__ = [
    "compress_df",
    "discrete_mf",
    "DiscretizedFrame",
    "DiscreteDesign",
    "build_discrete_design",
    "discrete_full_X",
    "Xbd",
    "XWXd",
    "XWyd",
]


# ---------------------------------------------------------------------------
# compress.df  — bam.r:122-184
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CompressResult:
    """Return value of :func:`compress_df`.

    ``xu`` is a dict matching the input ``dat`` layout: keys are the
    column names; values are the discretised unique values (1-D arrays).
    For matrix arguments the unique-pool is shared across all matrix
    columns of the input — values appear once each.

    ``k`` is the index mapping. Its shape is ``(n,)`` when *every*
    column of the input is 1-D, and ``(n, m_cols)`` when any column is a
    matrix (the matrix-argument summation case). Indices are 0-based;
    ``xu[name][k[i, q]]`` reproduces row ``i``, matrix-column ``q`` of
    the discretised input.
    """
    xu: dict[str, np.ndarray]
    k: np.ndarray


def compress_df(dat: dict[str, np.ndarray], m: Optional[int] = None,
                *, rng: Optional[RMersenneTwister] = None) -> _CompressResult:
    """Discretise a small dataframe by rounding (numeric) / dedup (factor).

    Direct port of mgcv ``compress.df`` (bam.r:122-184). The input
    ``dat`` is a dict of named columns sharing the same number of rows
    ``n``. Each column is one of:

      * 1-D numeric ``np.ndarray`` — rounded to a length-``m`` regular
        grid spanning ``[min, max]`` of the column.
      * 1-D factor-like (string / object dtype, or any ``np.ndarray`` of
        kind ``"O"``/``"U"``) — kept verbatim, dedup'd to its unique
        levels.
      * 2-D numeric ``np.ndarray`` of shape ``(n, m_cols)`` — under
        mgcv's summation convention, the entries are pooled into one
        unique table; the returned ``k`` is shaped ``(n, m_cols)`` so
        ``xu[name][k[i, q]]`` reproduces matrix entry ``[i, q]``.

    The default ``m`` follows mgcv: 1000 if a single variable, 100 in
    2-D, 25 in 3+. If supplied for a multi-variable input, mgcv reduces
    it to ``round(m**(1/d)) + 1`` so the joint grid stays bounded.

    A random shuffle of the unique-row order (mgcv bam.r:170-175) breaks
    spurious dependencies between jointly-discretised covariates, which
    would otherwise confuse the ``gam.side`` identifiability check. The
    caller is responsible for fixing the RNG state outside this routine
    (mgcv uses ``temp.seed(8547)`` in :func:`discrete_mf`); supply
    ``rng=`` to override. The default :class:`RMersenneTwister` is seeded
    to match ``discrete.mf``'s ``temp.seed(8547)``.
    """
    if rng is None:
        rng = RMersenneTwister(8547)
    names = list(dat.keys())
    d = len(names)
    n = next(iter(dat.values())).shape[0]
    if m is None:
        m = 1000 if d == 1 else (100 if d == 2 else 25)
    elif d > 1:
        # mgcv: m <- round(m^{1/d}) + 1
        m = int(round(m ** (1.0 / d))) + 1

    # Detect factor / matrix columns. mgcv treats string / object arrays
    # as factors; numeric (any float / int dtype) gets the rounding path.
    is_factor = {nm: _is_factor_arr(dat[nm]) for nm in names}
    is_matrix = {nm: dat[nm].ndim == 2 for nm in names}

    # mgcv ``mm`` (metric grid points) and ``mf`` (factor grid points)
    # — used as the cap above which rounding kicks in. Factor cols
    # contribute their level count; numeric cols contribute ``m``.
    mf_total = 1
    mm_total = 1
    for nm in names:
        if is_factor[nm]:
            mf_total *= int(np.unique(np.asarray(dat[nm]).ravel()).size)
        else:
            mm_total *= m

    # mgcv: if the first column is a matrix, all columns are vectorised
    # (matrix-arg case). Build a working dict of 1-D arrays of length n*m_cols.
    matrix_input = is_matrix[names[0]]
    if matrix_input:
        ncols_mat = dat[names[0]].shape[1]
        flat: dict[str, np.ndarray] = {}
        for nm in names:
            arr = dat[nm]
            if arr.ndim == 1:
                # broadcast scalar columns across the matrix-column axis
                arr = np.broadcast_to(arr[:, None], (n, ncols_mat))
            flat[nm] = arr.reshape(-1)
        work = flat
        n_eff = n * ncols_mat
    else:
        work = {nm: np.asarray(dat[nm]).ravel() for nm in names}
        n_eff = n

    # Initial uniquecombs on raw (or vectorised) input.
    xu_table, k_idx = _uniquecombs(work, names)

    if xu_table[names[0]].size > mm_total * mf_total:
        # Too many unique combinations — round metric variables to an
        # m-point grid before re-deduplicating (mgcv bam.r:155-163).
        rounded = {}
        for nm in names:
            if is_factor[nm]:
                rounded[nm] = work[nm]
            else:
                col = work[nm].astype(float)
                xl_lo = float(np.min(col))
                xl_hi = float(np.max(col))
                if xl_hi == xl_lo:
                    rounded[nm] = col.copy()
                else:
                    grid = np.linspace(xl_lo, xl_hi, m)
                    dx = grid[1] - grid[0]
                    kx = np.round((col - xl_lo) / dx).astype(int)
                    rounded[nm] = grid[kx]
        work = rounded
        xu_table, k_idx = _uniquecombs(work, names)

    nu = xu_table[names[0]].size

    if nu == n_eff:
        # No compression possible — return original ordering with identity index.
        k_out = np.arange(n_eff, dtype=np.int64)
        if matrix_input:
            k_out = k_out.reshape(n, ncols_mat)
        return _CompressResult(xu={nm: work[nm].copy() for nm in names}, k=k_out)

    # Shuffle xu rows to break induced dependencies (bam.r:171).
    perm = rng.sample_no_replace(nu, nu)
    xu_table = {nm: xu_table[nm][np.argsort(perm)] for nm in names}
    # ``perm[old_pos] = new_pos``; old k pointed to old_pos, after the
    # shuffle the same data should point to new_pos.
    k_idx = perm[k_idx]

    if matrix_input:
        k_idx = k_idx.reshape(n, ncols_mat)

    return _CompressResult(xu=xu_table, k=k_idx.astype(np.int64))


def _is_factor_arr(a: np.ndarray) -> bool:
    """Treat string / unicode / object arrays as factors."""
    return a.dtype.kind in ("U", "O", "S")


def _uniquecombs(work: dict[str, np.ndarray],
                 names: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Numpy port of R's ``uniquecombs`` (single-thread).

    Returns ``(xu, idx)`` where ``xu`` is a dict of unique columns (in
    canonical sort order) and ``idx[i]`` is the unique-row index for
    input row ``i``.
    """
    n = next(iter(work.values())).size
    if len(names) == 1:
        col = work[names[0]]
        u, inv = np.unique(col, return_inverse=True)
        return {names[0]: u}, inv.astype(np.int64)
    # Multi-column: stack into a structured key. Numeric columns are kept
    # numeric; factor columns are converted to integer codes with the
    # same lex order as ``np.unique``.
    keys: list[np.ndarray] = []
    for nm in names:
        col = work[nm]
        if _is_factor_arr(col):
            _, codes = np.unique(col, return_inverse=True)
            keys.append(codes)
        else:
            keys.append(col)
    # Use a structured array to do row-wise uniquecombs.
    dtype = np.dtype([(nm, k.dtype) for nm, k in zip(names, keys)])
    arr = np.empty(n, dtype=dtype)
    for nm, k in zip(names, keys):
        arr[nm] = k
    u, inv = np.unique(arr, return_inverse=True)
    xu = {nm: u[nm] for nm in names}
    # For factor cols we stored codes; the unique table here returns
    # codes in lex order — but the original column had string values.
    # We need to map codes back to the original strings.
    for nm in names:
        col = work[nm]
        if _is_factor_arr(col):
            levels, _ = np.unique(col, return_inverse=True)
            xu[nm] = levels[xu[nm]]
    return xu, inv.astype(np.int64)


# ---------------------------------------------------------------------------
# discrete.mf  — bam.r:201-380
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiscretizedFrame:
    """Output of :func:`discrete_mf`. Compact representation of an
    arbitrary mgcv model frame after per-marginal discretisation.

    Attributes
    ----------
    mf : dict[str, np.ndarray]
        Discretised model frame. Each named entry is a 1-D array; values
        are padded to the longest unique-table length (``maxr`` in mgcv)
        with random repeats so all entries share length, matching
        ``mgcv::discrete.mf`` (bam.r:355-365). Padding rows are *never*
        referenced via ``k`` — they exist only so downstream code can
        treat ``mf`` as a regular frame.
    k : np.ndarray
        Integer index matrix of shape ``(n, n_k_cols)``. ``k[i, q]`` is
        the row-index into the discretised table for variable
        ``names_for_col(q)`` of original observation ``i``. ``n_k_cols``
        sums over scalar variables (one column each) plus matrix
        variables (one column per matrix column under the summation
        convention). The final column is the all-ones intercept index.
    ks : np.ndarray
        ``(n_vars + 1, 2)`` int matrix. ``ks[j, 0]:ks[j, 1]`` is the
        slice of ``k`` columns associated with the ``j``th variable in
        ``names``. Final row corresponds to the intercept.
    nr : np.ndarray
        ``(n_vars + 1,)`` int vector. ``nr[j]`` is the number of unique
        rows for variable ``j`` (length of the un-padded ``mf`` entry).
        Final entry is 1 for the intercept.
    names : list[str]
        Variable names matching ``ks``/``nr``, in the order they were
        first discretised. ``"(Intercept)"`` is appended last.
    n : int
        Original (un-discretised) row count.
    """
    mf: dict[str, np.ndarray]
    k: np.ndarray
    ks: np.ndarray
    nr: np.ndarray
    names: list[str]
    n: int


def check_term(term: Sequence[str], rec: dict) -> int:
    """Has any variable in ``term`` already been discretised?

    Direct port of mgcv ``check.term`` (bam.r:185-200). Returns the
    1-based index of the prior discretisation if every variable in
    ``term`` matches an existing discretisation of the same dimension;
    raises :class:`ValueError` for partial overlap (mgcv: "bam can not
    discretize with this nesting structure"); returns 0 if no overlap.
    """
    vnames = rec["vnames"]
    ki = rec["ki"]
    d = rec["d"]
    ii = [j for j, nm in enumerate(vnames) if nm in term]
    if ii:
        i_min = min(ii)
        if len(term) == d[i_min]:
            if any(t not in [vnames[j] for j in ii] for t in term):
                raise ValueError(
                    "bam can not discretize with this nesting structure"
                )
            return ki[i_min]
        raise ValueError("bam can not discretize with this nesting structure")
    return 0


def discrete_mf(smooth_specs: list[dict], mf: pl.DataFrame,
                names_pmf: Sequence[str], m: Optional[int] = None,
                *, rng: Optional[RMersenneTwister] = None,
                full: bool = True) -> DiscretizedFrame:
    """Discretise the model frame ``mf`` per marginal of every smooth.

    Direct port of mgcv ``discrete.mf`` (bam.r:201-380). Walks the
    ``smooth_specs`` list (one dict per smooth, with keys ``term`` —
    list[str] of variables, ``by`` — str/None, ``margins`` — list of
    margin specs each with its own ``term``), discretises each
    marginal's variables jointly, and assembles:

      * ``mf``  — discretised, padded to common length ``maxr``
      * ``k``   — index matrix
      * ``ks``  — per-variable ``k``-column ranges
      * ``nr``  — per-variable un-padded length

    Followed by the parametric covariates (``names_pmf``), each
    discretised individually, and finally an intercept index column.

    ``rng`` should be supplied with a fixed seed for reproducibility.
    mgcv uses ``temp.seed(8547)`` (bam.r:233) — the default
    ``RMersenneTwister(8547)`` matches that exactly.
    """
    if rng is None:
        rng = RMersenneTwister(8547)

    n = mf.height
    # Pre-count how many index columns ``k`` will need: each smooth term
    # contributes ``len(margins) + (by != None)`` marginals; each parametric
    # variable contributes 1.
    nk = 0
    for spec in smooth_specs:
        n_marg = len(spec.get("margins", [{"term": spec["term"]}]))
        nk += n_marg + (1 if spec.get("by") not in (None, "NA") else 0)
    pmf_in_mf = [nm for nm in names_pmf if nm in mf.columns]
    nk += len(pmf_in_mf)

    # Bookkeeping. ``k`` will grow with extra columns when matrix-arg
    # smooths are encountered (each matrix column becomes one ``k`` column).
    k = np.zeros((n, nk), dtype=np.int64)
    ks = np.full((nk, 2), -1, dtype=np.int64)
    nr = np.zeros(nk, dtype=np.int64)
    var_order: list[str] = []
    mf0: dict[str, np.ndarray] = {}
    rec = {"vnames": [], "ki": [], "d": []}
    ik = -1  # 0-based marginal index counter (mgcv ``ik`` is 1-based)

    # Walk smooths, discretising each marginal once.
    def _discretise_marginal(termi: list[str], mi: Optional[int]):
        nonlocal ik
        prev = check_term(termi, rec)
        if prev != 0:
            return  # already discretised — re-use the entry
        ik += 1
        # Pull out the columns referenced in ``termi`` from the model frame.
        dat: dict[str, np.ndarray] = {}
        for nm in termi:
            s = mf[nm]
            if is_matrix_col(s):
                dat[nm] = matrix_to_2d(s)
            else:
                dat[nm] = s.to_numpy()
        cr = compress_df(dat, m=mi, rng=rng)
        ki = cr.k                                # (n,) or (n, m_cols)
        if ki.ndim == 1:
            ks[ik, 0] = ks[ik - 1, 1] if ik > 0 else 0
            ks[ik, 1] = ks[ik, 0] + 1
            k[:, ks[ik, 0]] = ki
        else:
            ks[ik, 0] = ks[ik - 1, 1] if ik > 0 else 0
            ks[ik, 1] = ks[ik, 0] + ki.shape[1]
            # Extend k if needed.
            need_cols = ks[ik, 1] - k.shape[1]
            if need_cols > 0:
                k_ext = np.zeros((n, need_cols), dtype=np.int64)
                # ``k`` is captured in closure; rebind via mutation
                k_full = np.concatenate([k, k_ext], axis=1)
                # Replace the whole column block.
                _set_k(k_full)
                k_local = k_full
            else:
                k_local = k
            k_local[:, ks[ik, 0]:ks[ik, 1]] = ki
        nr[ik] = cr.xu[termi[0]].size
        # Take the first variable's column as the canonical mf entry name.
        # Each variable in termi maps to its own discretised column in mf0
        # but they share one set of (k, ks, nr) — mgcv does this via
        # duplicated rows in nr/ks, so we replicate that here.
        nr_first = nr[ik]
        ks_first = ks[ik].copy()
        var_order.append(termi[0])
        mf0[termi[0]] = cr.xu[termi[0]]
        # Duplicate index info for every additional variable in this
        # joint discretisation (bam.r:255-262).
        for extra in termi[1:]:
            ik += 1
            ks[ik] = ks_first
            nr[ik] = nr_first
            var_order.append(extra)
            mf0[extra] = cr.xu[extra]
        # Update the dedup record.
        rec["vnames"].extend(termi)
        rec["ki"].extend([ik - len(termi) + 1] * len(termi))
        rec["d"].extend([len(termi)] * len(termi))

    # Helper for the rare matrix-arg ``k`` extension above.
    def _set_k(k_new: np.ndarray):
        nonlocal k
        k = k_new

    # --- smooths ---
    for spec in smooth_specs:
        margins = spec.get("margins", [{"term": spec["term"]}])
        by = spec.get("by")
        # ``by`` is processed first (matches mgcv jj==1 path).
        if by not in (None, "NA"):
            _discretise_marginal([by], m)
        for marg in margins:
            _discretise_marginal(list(marg["term"]), m)

    # --- parametric ---
    for nm in pmf_in_mf:
        # Skip if already discretised (shouldn't happen normally — para
        # vars don't appear in smooths — but mgcv guards anyway).
        if check_term([nm], rec) != 0:
            continue
        ik += 1
        s = mf[nm]
        if is_matrix_col(s):
            arr = matrix_to_2d(s)
            cr = compress_df({nm: arr}, m=m, rng=rng)
            mf_entry = cr.xu[nm]
            ki = cr.k.ravel()  # parametric matrix is dropped to vector
            nr[ik] = mf_entry.size
        else:
            arr = s.to_numpy()
            cr = compress_df({nm: arr}, m=m, rng=rng)
            mf_entry = cr.xu[nm]
            ki = cr.k
            nr[ik] = mf_entry.size
        ks[ik, 0] = ks[ik - 1, 1] if ik > 0 else 0
        ks[ik, 1] = ks[ik, 0] + 1
        k[:, ks[ik, 0]] = ki
        var_order.append(nm)
        mf0[nm] = mf_entry

    # --- pad mf0 to common length ---
    if full and mf0:
        maxr = max(arr.size for arr in mf0.values())
        for nm, arr in list(mf0.items()):
            if arr.size < maxr:
                # mgcv: ``mf0[[i]][(me+1):maxr] <- sample(mf0[[i]], maxr-me, replace=TRUE)``
                pad = arr[rng.sample_replace(arr.size, maxr - arr.size)]
                mf0[nm] = np.concatenate([arr, pad])
    else:
        maxr = max((arr.size for arr in mf0.values()), default=0)

    # --- intercept ---
    ik += 1
    # Trim k to the columns actually used.
    used_cols = int(np.max(ks[:ik]) if ik > 0 else 0)
    if used_cols < k.shape[1]:
        k = k[:, :used_cols]
    elif used_cols > k.shape[1]:
        # k was extended for matrix args; ensure size matches.
        pass
    # Append the intercept column (all 0 → unique row 0 = the constant 1).
    k = np.concatenate([k, np.zeros((n, 1), dtype=np.int64)], axis=1)
    ks_final = np.concatenate(
        [ks[:ik], np.array([[k.shape[1] - 1, k.shape[1]]], dtype=np.int64)],
        axis=0,
    )
    nr_final = np.concatenate([nr[:ik], np.array([1], dtype=np.int64)])
    var_order.append("(Intercept)")
    mf0["(Intercept)"] = np.ones(1, dtype=float)

    return DiscretizedFrame(
        mf=mf0, k=k, ks=ks_final, nr=nr_final, names=var_order, n=n,
    )


# ---------------------------------------------------------------------------
# Discrete design — per-marginal Xd blocks plus term packing.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _DiscreteTerm:
    """Per-term entry in :class:`DiscreteDesign`.

    A "term" here is one row in mgcv's ``ts``/``dt`` structure: one
    block of contiguous columns in the global coefficient vector.
    Parametric blocks have ``kind="param"`` and a single ``Xd``
    referenced by a single ``k`` column. Smooth blocks have one
    ``Xd`` per marginal with column-index entries ``k_cols`` aligning
    to the global ``k[:, ks[var, 0]:ks[var, 1]]`` slice for each
    marginal.
    """
    kind: str                        # "param" | "single" | "tensor"
    Xd_list: list[np.ndarray]        # one (m_j, p_j) block per marginal
    k_cols: list[tuple[int, int]]    # (start, stop) into global k for each marginal
    coef_slice: slice                # where this term lives in the full coef
    qc: int = 0                      # tensor-constraint indicator (1 if Householder)
    v: Optional[np.ndarray] = None   # Householder vec, length = Π p_j (qc=1 case)
    # absorb / by / keep_cols replays for the *whole* term — applied as
    # column transforms on the (Khatri-Rao) tensor block. None for params
    # and unconstrained smooths.
    absorb: Optional[object] = None
    by: Optional[object] = None
    keep_cols: Optional[np.ndarray] = None
    # Predict-time replay (used for predict.bamd, not the fitter).
    spec: Optional[BasisSpec] = None
    label: str = ""


@dataclass(slots=True)
class DiscreteDesign:
    """Compressed design store consumed by the discrete kernels.

    Mirrors mgcv's ``G$Xd``/``G$k``/``G$ks``/``G$ts``/``G$dt``/``G$v``/
    ``G$qc``/``G$drop`` package (bam.r:2300+ in ``bam`` setup), but
    represented as a Python dataclass with one :class:`_DiscreteTerm`
    per term. Total coef count is ``p`` (sum over terms of their post-
    constraint column count).
    """
    terms: list[_DiscreteTerm]
    k: np.ndarray                    # (n, n_k_cols) global index matrix
    ks: np.ndarray                   # (n_vars+1, 2) per-variable k-column slice
    nr: np.ndarray                   # (n_vars+1,) per-variable unique-row count
    n: int                           # original observation count
    p: int                           # total coef count (post-constraint)
    var_index: dict[str, int]        # variable name → row in ks/nr
    # Lazily populated caches — invariant under PIRLS (only depend on
    # the design's Xd_list / k / constraint structure, not on weights
    # or coefs). Set to ``False`` to disable caching for very-large-n
    # cases where ``X_full`` doesn't fit in memory.
    _full_X_cache: object = None
    _T_cache: object = None


def build_discrete_design(blocks: list[SmoothBlock],
                          X_param_full: np.ndarray,
                          dframe: DiscretizedFrame,
                          *,
                          param_terms: Sequence[str] = ("(Intercept)",),
                          ) -> DiscreteDesign:
    """Build :class:`DiscreteDesign` from a fitted set of
    :class:`SmoothBlock` plus a discretised model frame.

    The basis machinery for each smooth (knots, eigenbasis, by-mask,
    absorb constraint) has *already* been frozen in ``block.spec`` —
    typically by running ``materialize_smooths`` on a representative
    subsample (mgcv's ``mini.mf`` flow). What this routine does:

      * For each smooth, evaluate the per-marginal raw bases at the
        unique discretised values (one tiny ``predict_mat``-style call
        per margin), giving a list of ``Xd_j`` of shape ``(nr_j, p_j_raw)``.
      * Capture the term's ``by`` / ``absorb`` / ``keep_cols`` so the
        kernels can apply them on the row-tensor product at compute time.
      * For parametric columns, store the un-discretised columns of
        ``X_param_full`` directly (no compression) — equivalent to a
        single-marginal "term" with ``Xd = X_param_full`` and ``k =
        identity``.

    Constraint Householder vectors (mgcv ``v``/``qc``) are *not* yet
    extracted from ``absorb`` — for now we apply absorb post-hoc as a
    column transform on the materialised row-tensor. A future
    optimisation can switch to Householder for ``te`` smooths.
    """
    var_index = {nm: j for j, nm in enumerate(dframe.names)}
    terms: list[_DiscreteTerm] = []
    p_total = 0

    # Parametric columns: treat each parametric column-group as a single
    # "term". For now we keep them as one big block — index into
    # X_param_full via a synthetic identity ``k`` column. The simplest
    # representation just stores the full param matrix as a single Xd
    # with k-col = -1 (signal: identity gather). The kernels handle this
    # via a special-case branch.
    if X_param_full is not None and X_param_full.shape[1] > 0:
        p_par = X_param_full.shape[1]
        terms.append(_DiscreteTerm(
            kind="param",
            Xd_list=[np.asarray(X_param_full, dtype=float)],
            k_cols=[(-1, -1)],     # sentinel: gather is identity
            coef_slice=slice(p_total, p_total + p_par),
            label="(parametric)",
        ))
        p_total += p_par

    # Discretised model frame as a polars DataFrame with each column
    # length = nr[var] (no padding here — we feed only the unique values
    # so basis evaluators see the right domain).
    for block in blocks:
        spec = block.spec
        if spec is None:
            raise ValueError(
                f"SmoothBlock {block.label!r} has no spec — predict_mat "
                "replay is required for the discrete fitter"
            )
        term_vars = list(block.term)
        # Identify margins: tensor smooths have ``raw`` of type
        # ``_TensorRawBasis`` / ``_T2RawBasis`` / ``_T2PredictRawBasis``;
        # everything else is single-margin. Each margin variable list comes
        # from the raw basis itself (``_raw_basis_vars``) — that's the only
        # source that survives mgcv's ``tero`` reorder, where the block's
        # declaration order (``block.term``) no longer matches the post-
        # tero margin order.
        raw = spec.predict_raw if spec.predict_raw is not None else spec.raw
        if isinstance(raw, _TensorRawBasis):
            margin_raws = list(raw.margins)
            margin_vars = [_raw_basis_vars(m) or term_vars for m in margin_raws]
        elif isinstance(raw, (_T2RawBasis, _T2PredictRawBasis)):
            margin_raws = list(raw.margins)
            margin_vars = [_raw_basis_vars(m) or term_vars for m in margin_raws]
        else:
            margin_raws = [raw]
            margin_vars = [_raw_basis_vars(raw) or term_vars]

        # Evaluate each marginal raw basis on the discretised unique
        # values for its variables. The frame for marginal j is built
        # from ``dframe.mf`` taking exactly the variables in
        # ``margin_vars[j]``, length ``nr[var0]``.
        Xd_list: list[np.ndarray] = []
        k_cols: list[tuple[int, int]] = []
        for mvars, mraw in zip(margin_vars, margin_raws):
            v0 = mvars[0]
            j_var = var_index[v0]
            length_j = int(dframe.nr[j_var])
            sub = {nm: dframe.mf[nm][:length_j] for nm in mvars}
            sub_df = pl.DataFrame(sub)
            Xd = mraw.eval(sub_df)
            Xd_list.append(np.asarray(Xd, dtype=float))
            k_cols.append((int(dframe.ks[j_var, 0]), int(dframe.ks[j_var, 1])))

        # Term column count after by/absorb/keep_cols. Use block.X.shape[1]
        # as the authoritative post-transform width.
        p_term = block.X.shape[1]
        kind = "single" if len(margin_raws) == 1 else "tensor"
        terms.append(_DiscreteTerm(
            kind=kind,
            Xd_list=Xd_list,
            k_cols=k_cols,
            coef_slice=slice(p_total, p_total + p_term),
            absorb=spec.absorb,
            by=spec.by,
            keep_cols=spec.keep_cols,
            spec=spec,
            label=block.label,
        ))
        p_total += p_term

    return DiscreteDesign(
        terms=terms, k=dframe.k, ks=dframe.ks, nr=dframe.nr,
        n=dframe.n, p=p_total, var_index=var_index,
    )


def _raw_basis_vars(raw: _RawBasis) -> list[str]:
    """Return the variable names a raw basis evaluates on.

    Walks past ``_LinearTransformRawBasis`` wrappers (which do not declare
    their own ``term`` — they inherit from the inner basis). For mgcv-style
    leaf classes (``_CRRawBasis``, ``_TPRawBasis``, etc.) ``term`` is either
    a single string (1-D) or a list of strings (multi-D). Returns a list
    in either case so callers can iterate uniformly.
    """
    inner = raw
    while isinstance(inner, _LinearTransformRawBasis):
        inner = inner.inner
    term_attr = getattr(inner, "term", None)
    if term_attr is None:
        return []
    if isinstance(term_attr, str):
        return [term_attr]
    return list(term_attr)


def _split_term_vars_by_margins(term_vars: list[str],
                                margin_raws: list[_RawBasis]) -> list[list[str]]:
    """Best-effort decomposition of ``term_vars`` across margin bases.

    For multi-d margins the raw basis carries no explicit variable list,
    so we fall back to splitting ``term_vars`` evenly: each margin gets
    the next chunk in declaration order. This matches mgcv's
    convention: ``te(x, y, z, d=c(2,1))`` gives margins
    ``[("x","y"), ("z",)]``.
    """
    n_marg = len(margin_raws)
    if n_marg == 1:
        return [list(term_vars)]
    # Heuristic: split evenly. For most te smooths each margin is 1-D.
    if len(term_vars) == n_marg:
        return [[v] for v in term_vars]
    # Multi-d margins: walk margin raws looking for ``ranks``/``margins``
    # attrs that hint at per-margin dimensionality. For ``_TensorRawBasis``
    # we could check if each margin's basis itself is a tensor — but in
    # practice te()'s margins are univariate. Fall back to evenly
    # distributing extras to the first margin (matches mgcv's d=
    # ordering).
    chunks: list[list[str]] = []
    extra = len(term_vars) - n_marg
    cursor = 0
    for j in range(n_marg):
        size = 1 + (extra if j == 0 else 0)
        chunks.append(term_vars[cursor:cursor + size])
        cursor += size
        extra = 0
    return chunks


# ---------------------------------------------------------------------------
# Kernels — Xbd / XWXd / XWyd
# ---------------------------------------------------------------------------


def _term_full_design(term: _DiscreteTerm, k: np.ndarray, n: int) -> np.ndarray:
    """Materialise ``X_term`` (n × p_term_post) for one term.

    Used by the kernels as a generic fallback: gather per-marginal Xd,
    Khatri-Rao multiply, sum across summation columns, then apply
    ``by`` / ``absorb`` / ``keep_cols``. This is the *correct*
    semantics; the optimised path (Householder constraint, tensor-
    aware scatter-add for X'WX) replicates this row-by-row but avoids
    forming the n × p_raw block.

    For now the fallback is the only path — it still avoids the
    chunked-QR overhead because ``Xd_list`` is small and reusable
    across PIRLS iterations.
    """
    if term.kind == "param":
        return term.Xd_list[0]

    # Determine summation width (same across margins for a given term —
    # mgcv enforces this by jointly discretising matrix-arg margins).
    q_lo, q_hi = term.k_cols[0]
    n_sum = q_hi - q_lo

    # Khatri-Rao across margins, then sum over summation columns.
    X_full = None
    for q_off in range(n_sum):
        # For each margin, gather rows: Xd_j[k[:, ks_j_lo + q_off], :]
        Xq_blocks = []
        for Xd_j, (ks_lo, ks_hi) in zip(term.Xd_list, term.k_cols):
            kcol = k[:, ks_lo + q_off]
            Xq_blocks.append(Xd_j[kcol])
        # Khatri-Rao over the margin blocks.
        if len(Xq_blocks) == 1:
            Xq = Xq_blocks[0]
        else:
            Xq = Xq_blocks[0]
            for nxt in Xq_blocks[1:]:
                # Row tensor: (n, p1) ⊗_row (n, p2) → (n, p1*p2)
                pa = Xq.shape[1]
                pb = nxt.shape[1]
                Xq = (Xq[:, :, None] * nxt[:, None, :]).reshape(-1, pa * pb)
        if X_full is None:
            X_full = Xq
        else:
            X_full = X_full + Xq

    # Apply by / absorb / keep_cols — these are column transforms.
    if term.by is not None:
        # by.apply expects a polars.DataFrame; build a synthetic one
        # from the original (un-discretised) frame. The kernels operate
        # on the n original rows so the by-column lives in the input
        # data, not in dframe.mf. We can't reconstruct it here without
        # the original data, so we expect the caller to apply ``by``
        # outside (or pass the data via the kernel API). For now we
        # leave by application to the discrete-design builder.
        pass
    if term.absorb is not None:
        X_full = term.absorb.apply(X_full)
    if term.keep_cols is not None:
        X_full = X_full[:, term.keep_cols]
    return X_full


def discrete_full_X(design: DiscreteDesign) -> np.ndarray:
    """Materialise the full ``n × p`` design matrix from the compressed
    representation.

    Result is cached on the design object — basis values on the
    discretised unique grid (``Xd_list``) are fixed for the life of the
    design, so the materialisation is invariant across PIRLS iters and
    across outer-Newton steps. The cache is populated lazily on first
    call and reused thereafter; this turns every downstream
    ``Xbd``/``XWXd``/``XWyd`` invocation into a single BLAS matmul on
    cached ``X_full``, which dominates scatter-add for the
    moderate-``n`` × moderate-``p`` regime where chicago lives.

    For very large ``n`` where ``X_full`` doesn't fit in memory, callers
    can opt out of caching by setting ``design._full_X_cache = False``
    before invoking the kernels — they will then route through the
    scatter-add paths in ``Xbd``/``XWXd``/``XWyd`` directly.
    """
    cache = design._full_X_cache
    if isinstance(cache, np.ndarray):
        return cache
    n = design.n
    blocks = [_term_full_design(t, design.k, n) for t in design.terms]
    full = np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
    if cache is not False:                 # ``False`` disables caching
        design._full_X_cache = full
    return full


# ---------------------------------------------------------------------------
# Scatter-add kernel helpers
# ---------------------------------------------------------------------------


def _grouped_sum_axis0(idx: np.ndarray, values: np.ndarray,
                       n_groups: int) -> np.ndarray:
    """Sum rows of ``values`` grouped by integer keys ``idx``.

    Pure-numpy equivalent of ``np.add.at(out, idx, values)`` but uses
    argsort + ``np.add.reduceat`` so the inner accumulation runs at
    BLAS-equivalent speed. ``idx`` is a length-n int array; ``values``
    has shape ``(n, ...)``; the output has shape ``(n_groups, ...)``.

    For 1-D ``values`` this collapses to ``np.bincount``.
    """
    if values.ndim == 1:
        return np.bincount(idx, weights=values, minlength=n_groups)
    if values.shape[0] == 0:
        return np.zeros((n_groups,) + values.shape[1:], dtype=values.dtype)
    order = np.argsort(idx, kind='stable')
    sorted_idx = idx[order]
    sorted_vals = values[order]
    unique_groups, starts = np.unique(sorted_idx, return_index=True)
    sums = np.add.reduceat(sorted_vals, starts, axis=0)
    out = np.zeros((n_groups,) + values.shape[1:], dtype=sorted_vals.dtype)
    out[unique_groups] = sums
    return out


def _term_constraint_T(term: _DiscreteTerm) -> Optional[np.ndarray]:
    """Materialise the constraint matrix ``T`` (p_raw × p_post) such that
    ``X_term_post = X_term_raw @ T``.

    Returns ``None`` for the identity case (no absorb / keep_cols), so
    callers can short-circuit the multiplication. For tensor smooths
    the absorb is the rank-1 sum-to-zero Householder; for singletons
    it's the per-margin absorb chain. Both are realised here by
    feeding ``np.eye(p_raw)`` through ``term.absorb.apply`` — the same
    path ``_term_full_design`` would use, but applied once at term
    setup rather than per row.
    """
    if term.kind == "param":
        return None
    p_raw = int(np.prod([Xd.shape[1] for Xd in term.Xd_list]))
    if term.absorb is None and term.keep_cols is None:
        return None
    T = np.eye(p_raw, dtype=float)
    if term.absorb is not None:
        T = term.absorb.apply(T)
    if term.keep_cols is not None:
        T = T[:, term.keep_cols]
    return np.ascontiguousarray(T)


def _design_constraint_Ts(design: DiscreteDesign) -> list[Optional[np.ndarray]]:
    """Per-term constraint matrices, cached on the design object.

    The ``T`` matrices depend only on the design (not on weights or
    coefs), so we compute them once and reuse across every PIRLS
    iteration.
    """
    cache = design._T_cache
    if cache is not None:
        return cache
    Ts = [_term_constraint_T(t) for t in design.terms]
    design._T_cache = Ts
    return Ts


# ---------------------------------------------------------------------------
# Per-term kernels — operate on the *unconstrained* raw column space
# (``p_raw``). Constraint application via ``T`` is layered on top in
# the public Xbd / XWyd / XWXd.
# ---------------------------------------------------------------------------


def _term_Xb_raw(term: _DiscreteTerm, b_raw: np.ndarray,
                 k: np.ndarray, n: int) -> np.ndarray:
    """Compute ``X_term_raw @ b_raw`` (length ``n``).

    Direct port of mgcv ``singleXb`` / ``tensorXb`` (discrete.c:375-444),
    but with the inner C loops replaced by numpy gathers + einsum.
    Tensor terms keep the C kernel's structure: contract the *final*
    marginal first against ``b`` to form ``C`` (the m_d-rowed working
    matrix), then per-q gather pre-final marginals at row indices and
    contract with ``C`` row-wise.
    """
    if term.kind == "param":
        return term.Xd_list[0] @ b_raw

    if term.kind == "single":
        Xd = term.Xd_list[0]
        ks_lo, ks_hi = term.k_cols[0]
        tmp = Xd @ b_raw                       # (m,)
        result = tmp[k[:, ks_lo]].copy()
        for q in range(ks_lo + 1, ks_hi):
            result += tmp[k[:, q]]
        return result

    return _tensor_Xb_raw(term, b_raw, k, n)


def _tensor_Xb_raw(term: _DiscreteTerm, b_raw: np.ndarray,
                   k: np.ndarray, n: int) -> np.ndarray:
    Xd_list = term.Xd_list
    d = len(Xd_list)
    ps = tuple(Xd.shape[1] for Xd in Xd_list)
    B = np.asarray(b_raw, dtype=float).reshape(ps)

    # C[..., g] = Σ_{l_d} B[..., l_d] · Xd_d[g, l_d]  — final marginal applied
    Xd_d = Xd_list[-1]
    if d == 2:
        C = B @ Xd_d.T                          # (p1, m_d)
    elif d == 3:
        C = np.einsum('ijd,gd->ijg', B, Xd_d)   # (p1, p2, m_d)
    else:
        in_letters = "abcdefghij"[:d]
        out_letters = in_letters[:-1] + "G"
        C = np.einsum(f"{in_letters},G{in_letters[-1]}->{out_letters}",
                      B, Xd_d)

    ks_lo_list = [term.k_cols[j][0] for j in range(d)]
    n_sum = term.k_cols[0][1] - term.k_cols[0][0]
    result = np.zeros(n, dtype=float)
    for q in range(n_sum):
        k_per_marg = [k[:, ks_lo_list[j] + q] for j in range(d)]
        if d == 2:
            X1_at_row = Xd_list[0][k_per_marg[0]]    # (n, p1)
            C_gathered = C[:, k_per_marg[1]]          # (p1, n)
            result += np.einsum('rp,pr->r', X1_at_row, C_gathered)
        elif d == 3:
            X1_at_row = Xd_list[0][k_per_marg[0]]    # (n, p1)
            X2_at_row = Xd_list[1][k_per_marg[1]]    # (n, p2)
            C_gathered = C[:, :, k_per_marg[2]]       # (p1, p2, n)
            result += np.einsum('rp,rq,pqr->r',
                                X1_at_row, X2_at_row, C_gathered)
        else:
            in_letters = "abcdefghij"[:d-1]
            X_at_rows = [Xd_list[j][k_per_marg[j]] for j in range(d-1)]
            C_gathered = C[(slice(None),) * (d-1) + (k_per_marg[d-1],)]
            expr = (",".join(f"r{l}" for l in in_letters)
                    + "," + in_letters + "r" + "->r")
            result += np.einsum(expr, *X_at_rows, C_gathered)
    return result


def _term_Xty_raw(term: _DiscreteTerm, wy: np.ndarray,
                  k: np.ndarray, n: int) -> np.ndarray:
    """Compute ``X_term_raw.T @ wy`` (length ``p_raw``).

    Direct port of ``singleXty`` / ``tensorXty`` (discrete.c:329-373).
    Singleton: ``temp = bincount(k_q, wy)``; ``Xty += Xd.T @ temp``,
    accumulated per q. Tensor: form the d-D scatter-summed weight
    tensor ``W̄`` from ``(k_1_q, …, k_d_q)`` then einsum it against
    every marginal ``Xd_j`` to land in the (p_1×…×p_d) coefficient
    space — equivalent to mgcv's per-pre-final-col extraction +
    ``singleXty(M_d, work, …)`` but vectorised over all pre-final
    columns at once.
    """
    if term.kind == "param":
        return term.Xd_list[0].T @ wy

    if term.kind == "single":
        Xd = term.Xd_list[0]
        m = Xd.shape[0]
        ks_lo, ks_hi = term.k_cols[0]
        temp = np.bincount(k[:, ks_lo], weights=wy, minlength=m)
        for q in range(ks_lo + 1, ks_hi):
            temp += np.bincount(k[:, q], weights=wy, minlength=m)
        return Xd.T @ temp

    return _tensor_Xty_raw(term, wy, k, n)


def _tensor_Xty_raw(term: _DiscreteTerm, wy: np.ndarray,
                    k: np.ndarray, n: int) -> np.ndarray:
    Xd_list = term.Xd_list
    d = len(Xd_list)
    ms = tuple(Xd.shape[0] for Xd in Xd_list)
    M = int(np.prod(ms))

    ks_lo_list = [term.k_cols[j][0] for j in range(d)]
    n_sum = term.k_cols[0][1] - term.k_cols[0][0]

    # W̄[g_1,…,g_d] = Σ_{rows with all k_j_q[row]=g_j} wy[row]
    W_flat = np.zeros(M, dtype=float)
    for q in range(n_sum):
        flat_idx = np.zeros(n, dtype=np.int64)
        stride = 1
        for j in range(d - 1, -1, -1):
            flat_idx = flat_idx + k[:, ks_lo_list[j] + q] * stride
            stride *= ms[j]
        W_flat += np.bincount(flat_idx, weights=wy, minlength=M)
    W_total = W_flat.reshape(ms)

    if d == 2:
        result = np.einsum('ab,ai,bj->ij',
                           W_total, Xd_list[0], Xd_list[1])
    elif d == 3:
        result = np.einsum('abc,ai,bj,ck->ijk',
                           W_total, Xd_list[0], Xd_list[1], Xd_list[2])
    else:
        in_letters = "abcdefghij"[:d]
        out_letters = "ABCDEFGHIJ"[:d]
        operand_subs = [in_letters] + [in_letters[j] + out_letters[j]
                                        for j in range(d)]
        expr = ",".join(operand_subs) + "->" + out_letters
        result = np.einsum(expr, W_total, *Xd_list)
    return result.reshape(-1)


def _term_pair_XWX_raw(term_a: _DiscreteTerm, term_b: _DiscreteTerm,
                       w: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    """Compute the unconstrained ``X_a_raw.T @ diag(w) @ X_b_raw`` block
    of shape ``(p_a_raw, p_b_raw)``.

    Direct port of mgcv ``XWXijs`` (discrete.c:1073-1430) for the
    ``acc_w == 1`` branch (n > m_im·m_jm — by far the common case for
    our matrix-arg models). The ``r``/``c`` sub-block decomposition is
    fused: we build the (n × p_a_pre × p_b_pre) outer-product weight
    tensor, scatter into a (m_a_final · m_b_final, p_a_pre · p_b_pre)
    grouped table, then contract with both final-marginal bases via a
    single einsum — equivalent to the per-(r,c) loop but vectorised.
    """
    if term_a.kind == "param" and term_b.kind == "param":
        Xa = term_a.Xd_list[0]
        Xb = term_b.Xd_list[0]
        return Xa.T @ (w[:, None] * Xb)

    if term_a.kind == "param":
        Xa = term_a.Xd_list[0]
        p_a = Xa.shape[1]
        p_b_raw = int(np.prod([Xd.shape[1] for Xd in term_b.Xd_list]))
        result = np.empty((p_a, p_b_raw), dtype=float)
        for i in range(p_a):
            result[i, :] = _term_Xty_raw(term_b, w * Xa[:, i], k, n)
        return result

    if term_b.kind == "param":
        return _term_pair_XWX_raw(term_b, term_a, w, k, n).T

    if term_a.kind == "single" and term_b.kind == "single":
        return _single_single_XWX(term_a, term_b, w, k, n)

    return _general_XWX(term_a, term_b, w, k, n)


def _single_single_XWX(term_a: _DiscreteTerm, term_b: _DiscreteTerm,
                       w: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    Xd_a = term_a.Xd_list[0]
    Xd_b = term_b.Xd_list[0]
    m_a = Xd_a.shape[0]
    m_b = Xd_b.shape[0]
    ks_a_lo, ks_a_hi = term_a.k_cols[0]
    ks_b_lo, ks_b_hi = term_b.k_cols[0]

    W_total = np.zeros((m_a, m_b), dtype=float)
    for q_a in range(ks_a_hi - ks_a_lo):
        k_a = k[:, ks_a_lo + q_a]
        for q_b in range(ks_b_hi - ks_b_lo):
            k_b = k[:, ks_b_lo + q_b]
            flat_idx = k_a * m_b + k_b
            W_flat = np.bincount(flat_idx, weights=w, minlength=m_a * m_b)
            W_total += W_flat.reshape(m_a, m_b)
    return Xd_a.T @ W_total @ Xd_b


def _row_tensor_pre(Xd_list: list[np.ndarray],
                    k_per_marg: list[np.ndarray],
                    da: int, n: int) -> np.ndarray:
    """Row-tensor of the *pre-final* marginals at this q's indices.

    Returns shape ``(n, p_pre)`` where ``p_pre = ∏_{j<da-1} p_j``.
    For singleton terms (da == 1) returns ``(n, 1)`` ones — caller
    treats it as the empty pre-tensor.
    """
    if da == 1:
        return np.ones((n, 1), dtype=float)
    if da == 2:
        return Xd_list[0][k_per_marg[0]]
    if da == 3:
        X1 = Xd_list[0][k_per_marg[0]]                          # (n, p1)
        X2 = Xd_list[1][k_per_marg[1]]                          # (n, p2)
        return (X1[:, :, None] * X2[:, None, :]).reshape(n, -1)
    blocks = [Xd_list[j][k_per_marg[j]] for j in range(da - 1)]
    out = blocks[0]
    for jj in range(1, da - 1):
        out = (out[:, :, None] * blocks[jj][:, None, :]).reshape(n, -1)
    return out


def _general_XWX(term_a: _DiscreteTerm, term_b: _DiscreteTerm,
                 w: np.ndarray, k: np.ndarray, n: int) -> np.ndarray:
    Xd_a_list = term_a.Xd_list
    Xd_b_list = term_b.Xd_list
    da = len(Xd_a_list)
    db = len(Xd_b_list)
    pa = [Xd.shape[1] for Xd in Xd_a_list]
    pb = [Xd.shape[1] for Xd in Xd_b_list]
    p_a_raw = int(np.prod(pa))
    p_b_raw = int(np.prod(pb))
    p_a_pre = int(np.prod(pa[:-1])) if da > 1 else 1
    p_b_pre = int(np.prod(pb[:-1])) if db > 1 else 1
    p_a_final = pa[-1]
    p_b_final = pb[-1]
    Xd_a_final = Xd_a_list[-1]
    Xd_b_final = Xd_b_list[-1]
    m_a_final = Xd_a_final.shape[0]
    m_b_final = Xd_b_final.shape[0]

    ks_a_lo = [term_a.k_cols[j][0] for j in range(da)]
    ks_b_lo = [term_b.k_cols[j][0] for j in range(db)]
    n_sum_a = term_a.k_cols[0][1] - term_a.k_cols[0][0]
    n_sum_b = term_b.k_cols[0][1] - term_b.k_cols[0][0]

    XWX_block = np.zeros((p_a_pre, p_a_final, p_b_pre, p_b_final),
                          dtype=float)

    for q_a in range(n_sum_a):
        k_a_per = [k[:, ks_a_lo[j] + q_a] for j in range(da)]
        Xa_pre = _row_tensor_pre(Xd_a_list, k_a_per, da, n)    # (n, p_a_pre)
        k_a_final = k_a_per[-1]

        for q_b in range(n_sum_b):
            k_b_per = [k[:, ks_b_lo[j] + q_b] for j in range(db)]
            Xb_pre = _row_tensor_pre(Xd_b_list, k_b_per, db, n)
            k_b_final = k_b_per[-1]

            # Outer-product weight: F[row, r, c] = w[row] · Xa_pre[row,r] · Xb_pre[row,c]
            F = (w[:, None, None]
                 * Xa_pre[:, :, None] * Xb_pre[:, None, :]
                 ).reshape(n, p_a_pre * p_b_pre)

            joint_idx = (k_a_final.astype(np.int64) * m_b_final
                          + k_b_final.astype(np.int64))
            W_flat = _grouped_sum_axis0(joint_idx, F,
                                         m_a_final * m_b_final)
            W_grouped = W_flat.reshape(m_a_final, m_b_final,
                                        p_a_pre, p_b_pre)

            # Out[r, la, c, lb] = Σ_{ga, gb} Xd_af[ga,la] · W̄[ga,gb,r,c] · Xd_bf[gb,lb]
            XWX_block += np.einsum('Aa,ABrc,Bb->racb',
                                    Xd_a_final, W_grouped, Xd_b_final)

    return XWX_block.reshape(p_a_raw, p_b_raw)


# ---------------------------------------------------------------------------
# Public kernels
# ---------------------------------------------------------------------------


def Xbd(design: DiscreteDesign, beta: np.ndarray,
        *, X: Optional[np.ndarray] = None,
        use_kernel: bool = False) -> np.ndarray:
    """Compute ``X β`` on the compressed design.

    Direct port of mgcv ``Xbd`` (misc.r:385 + src/discrete.c:474-557).
    Default path uses the cached full-X matmul (set up by
    :func:`discrete_full_X`); set ``use_kernel=True`` to instead route
    through the per-term scatter-add kernels (lift
    ``β_post → β_raw = T·β_post``, then ``_term_Xb_raw`` accumulates
    into the n-vector η). The ``X=`` argument lets callers pass an
    explicit X matrix — used mainly as a verification oracle.
    """
    beta = np.asarray(beta, dtype=float)
    if X is not None:
        return X @ beta
    if not use_kernel:
        return discrete_full_X(design) @ beta
    n = design.n
    eta = np.zeros(n, dtype=float)
    Ts = _design_constraint_Ts(design)
    for term, T in zip(design.terms, Ts):
        b_post = beta[term.coef_slice]
        b_raw = b_post if T is None else (T @ b_post)
        eta += _term_Xb_raw(term, b_raw, design.k, n)
    return eta


def XWXd(design: DiscreteDesign, w: np.ndarray,
         *, X: Optional[np.ndarray] = None,
         use_kernel: bool = False) -> np.ndarray:
    """Compute ``X' diag(w) X`` on the compressed design.

    Direct port of mgcv ``XWXd0`` (discrete.c:1457 driver around
    ``XWXijs``). Default path uses the cached full-X (single
    ``X' diag(w) X`` BLAS gemm); set ``use_kernel=True`` to route
    through per-term-pair scatter-add blocks (each block computed via
    :func:`_term_pair_XWX_raw` in the unconstrained raw space, then
    sandwiched ``T_a' (·) T_b`` to land in the post-constraint Gram).
    The full Gram is symmetric so kernel mode only computes
    upper-triangular blocks and mirrors.
    """
    w_arr = np.asarray(w, dtype=float)
    if X is None and not use_kernel:
        X = discrete_full_X(design)
    if X is not None:
        sqrt_w = np.sqrt(w_arr)
        Xw = sqrt_w[:, None] * X
        return Xw.T @ Xw

    n = design.n
    p = design.p
    XWX = np.zeros((p, p), dtype=float)
    Ts = _design_constraint_Ts(design)
    terms = design.terms

    for i, term_i in enumerate(terms):
        T_i = Ts[i]
        s_i = term_i.coef_slice
        block_raw = _term_pair_XWX_raw(term_i, term_i, w_arr,
                                        design.k, n)
        if T_i is None:
            block_post = block_raw
        else:
            block_post = T_i.T @ block_raw @ T_i
        XWX[s_i, s_i] = block_post

        for j in range(i + 1, len(terms)):
            term_j = terms[j]
            T_j = Ts[j]
            s_j = term_j.coef_slice
            block_raw = _term_pair_XWX_raw(term_i, term_j, w_arr,
                                            design.k, n)
            if T_i is None and T_j is None:
                block_post = block_raw
            elif T_i is None:
                block_post = block_raw @ T_j
            elif T_j is None:
                block_post = T_i.T @ block_raw
            else:
                block_post = T_i.T @ block_raw @ T_j
            XWX[s_i, s_j] = block_post
            XWX[s_j, s_i] = block_post.T
    return XWX


def XWyd(design: DiscreteDesign, w: np.ndarray, y: np.ndarray,
         *, X: Optional[np.ndarray] = None,
         use_kernel: bool = False) -> np.ndarray:
    """Compute ``X' (w · y)`` on the compressed design.

    Direct port of mgcv ``XWyd`` (misc.r:333 + src/discrete.c:717-802).
    Default path uses the cached full-X (single ``X' · (w·y)`` BLAS
    gemv); set ``use_kernel=True`` to scatter-add ``w·y`` into the
    m-d-grouped weight tensor per term, then contract against every
    marginal ``Xd`` to land in raw coefficient space, then apply ``T'``
    to land in the post-constraint slot.
    """
    w_arr = np.asarray(w, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if X is None and not use_kernel:
        X = discrete_full_X(design)
    if X is not None:
        return X.T @ (w_arr * y_arr)

    n = design.n
    p = design.p
    wy = w_arr * y_arr
    Xy = np.zeros(p, dtype=float)
    Ts = _design_constraint_Ts(design)
    for term, T in zip(design.terms, Ts):
        Xty_raw = _term_Xty_raw(term, wy, design.k, n)
        if T is None:
            Xy[term.coef_slice] = Xty_raw
        else:
            Xy[term.coef_slice] = T.T @ Xty_raw
    return Xy
