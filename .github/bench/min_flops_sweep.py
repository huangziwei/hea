"""Where does the vendor BLAS start beating hea's own kernels on this ISA?

`sparse::blas::MIN_FLOPS` is the per-call flop count above which a dense kernel
is handed to the vendor. It is `1.0e5`, and **every measurement behind that
number was taken on aarch64** — where hea's kernels compile to a baseline that
has NEON and a fused multiply-add, and where the vendor is Accelerate, whose
coprocessor round trip is half of why the cutoff is not zero.

Linux and Windows ship that constant with neither of those things true. Both
sides of the ratio move on x86-64 and in the same direction: hea's kernels get
worse (the target baseline is SSE2 with no FMA — not an oversight, but
`nmath::util::rfma`'s R-parity contract, which forbids a blanket
`target-feature=+fma`), and OpenBLAS gets better (AVX2/AVX-512, dispatched at
run time by `DYNAMIC_ARCH`). So the x86-64 crossing is somewhere below `1.0e5`,
plausibly at zero. This script is the receipt.

**It does not need CHOLMOD, and that is the point.** The question is not "is hea
faster than the reference" but "for a call of F flops, which of hea's own two
paths is faster" — so the instrument is hea against hea, with only the cutoff
moving. Nothing external to link, nothing to install, no corpus to fetch.

Requires a `blas-sweep` build, which reads the cutoff from the environment
instead of compiling it in, so seven columns cost one build rather than seven:

    maturin develop --release --features blas-sweep,blas-required,blas-openblas
    python .github/bench/min_flops_sweep.py

`SWEEP_N` overrides the grid sizes, `SWEEP_REPS` the best-of.
"""

import json
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sp

# kflop, i.e. the value of `MIN_FLOPS` in thousands. `inf` is the control: no
# routing at all, which is what the kernels did before the vendor was bound.
# `0` hands the vendor every call. The shipped constant is 100.
CUTOFFS = ("inf", "1000", "250", "100", "50", "25", "0")

# 2D grid Laplacians. The real corpus (pygridfit's LHS, pywarper's AtA) is not
# in the repo -- `dev/` is ignored -- and does not need to be: a cutoff is a
# property of the CALL SHAPES a factorization issues, and a banded 2D grid
# issues the same spread of small-to-middling supernodes these consumers do.
# Sizes bracket the corpus: 110², 220², 320².
SIZES = tuple(int(v) for v in os.environ.get("SWEEP_N", "110,220,320").split(","))
REPS = int(os.environ.get("SWEEP_REPS", "5"))
NRHS = (1, 4, 16, 32)


def laplacian(k):
    """The 5-point Laplacian on a `k x k` grid, SPD, upper triangle, CSC."""
    d = sp.diags_array(
        [np.ones(k - 1), np.full(k, -4.0), np.ones(k - 1)], offsets=[-1, 0, 1]
    )
    i = sp.eye_array(k)
    a = sp.kron(i, d) + sp.kron(sp.diags_array([np.ones(k - 1)], offsets=[1]), i)
    a = a + a.T
    n = k * k
    # Diagonally dominant, so it is SPD without a shift search.
    a = (-a + sp.eye_array(n) * 9.0).tocsc()
    return sp.csc_array(sp.triu(a).tocsc())


def child():
    """One cutoff, in its own process, because the cutoff is read once."""
    from hea import _rs
    from hea.sparse import build_info

    info = build_info()
    got = {"backend": info["backend"], "min_flops": info["min_flops"], "rows": []}
    for k in SIZES:
        a = laplacian(k)
        a.sort_indices()
        n = a.shape[0]
        ip = a.indptr.astype(np.int64)
        ii = a.indices.astype(np.int64)
        ax = a.data.astype(np.float64)
        arg = (n, ip, ii, ax, 1, 0.0, "amd")
        _rs.super_factorize(*arg, numeric_reps=0)  # warm the pool and the loader
        fac = _rs.super_factorize(*arg, numeric_reps=REPS)["numeric_ms"]
        solves = []
        for nrhs in NRHS:
            # `b` is one flat column-major block of `n * nrhs`, not an (n, nrhs)
            # array -- `super_solve` takes a `PyReadonlyArray1` and slices it.
            #
            # `solve_reps` is the whole reason to call it this way: the entry
            # point re-analyzes and refactorizes on every call, so timing the
            # call from Python would measure the analysis. `solve_ms` is the
            # best of `solve_reps` solves against one factor, which is the
            # quantity the eight solve kernels actually live in.
            b = np.ones(n * nrhs)
            sol = _rs.super_solve(n, ip, ii, ax, b, nrhs, 1, "A", 0.0, "amd", REPS)
            solves.append(sol["solve_ms"])
        got["rows"].append({"k": k, "n": n, "factorize": fac, "solve": solves})
    print(json.dumps(got))


def geomean(xs):
    return float(np.exp(np.mean(np.log(xs))))


def main():
    print(f"grid Laplacians {SIZES}, best-of-{REPS}, nrhs {NRHS}")
    print("one process per cutoff; ms, lower is better\n")
    out = {}
    for c in CUTOFFS:
        env = dict(os.environ)
        # `inf` is the control: no call ever clears it, so nothing is routed.
        env["HEA_BLAS_MIN_FLOPS"] = "1e300" if c == "inf" else str(int(c) * 1000)
        r = subprocess.run(
            [sys.executable, __file__, "--child"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if r.returncode:
            # Do not make a CI failure require a second run to diagnose.
            sys.exit(f"cutoff {c} failed:\n{r.stdout}\n{r.stderr}")
        out[c] = json.loads(r.stdout.strip().splitlines()[-1])
        eff = out[c]["min_flops"]
        want = float(env["HEA_BLAS_MIN_FLOPS"])
        if eff != want:
            # A build without `blas-sweep` compiles the cutoff in and ignores
            # the environment, which would silently make all seven columns the
            # same measurement.
            sys.exit(
                f"cutoff {c}: build reports min_flops={eff}, asked for {want}. "
                "Rebuild with --features blas-sweep."
            )
    print(f"backend: {out[CUTOFFS[0]]['backend']}\n")

    for i, k in enumerate(SIZES):
        print(f"{k}² (n = {out[CUTOFFS[0]]['rows'][i]['n']})")
        print(f"{'cutoff, kflop':<16}" + "".join(f"{c:>9}" for c in CUTOFFS))
        cells = "".join(f"{out[c]['rows'][i]['factorize']:>9.2f}" for c in CUTOFFS)
        print(f"{'factorize':<16}{cells}")
        for j, nrhs in enumerate(NRHS):
            cells = "".join(f"{out[c]['rows'][i]['solve'][j]:>9.3f}" for c in CUTOFFS)
            print(f"{'solve nrhs=' + str(nrhs):<16}{cells}")
        print()

    # One number per column: the geometric mean over every measurement, against
    # the no-routing control. Above 1 means routing at that cutoff is a win.
    print("geometric mean over all sizes and nrhs, x the `inf` control")
    print(f"{'':<16}" + "".join(f"{c:>9}" for c in CUTOFFS))
    ratios = {}
    for c in CUTOFFS:
        rs = []
        for i, _ in enumerate(SIZES):
            ctl, cur = out["inf"]["rows"][i], out[c]["rows"][i]
            rs.append(ctl["factorize"] / max(cur["factorize"], 1e-12))
            rs += [a / max(b, 1e-12) for a, b in zip(ctl["solve"], cur["solve"])]
        ratios[c] = geomean(rs)
    print(f"{'x control':<16}" + "".join(f"{ratios[c]:>9.3f}" for c in CUTOFFS))
    best = max(ratios, key=ratios.get)
    print(f"\nbest column: {best} kflop at {ratios[best]:.3f}x")
    print(f"shipped constant is 100 kflop, reading {ratios['100']:.3f}x")
    print(
        "\nRead the flat bottom, not the argmax: on aarch64 the 25-100 basin is "
        "one flat floor whose spread is under the instrument's own noise. What "
        "this run settles is whether x86-64's floor sits at 0 -- i.e. hand the "
        "vendor everything -- or somewhere above it."
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child()
    else:
        main()
