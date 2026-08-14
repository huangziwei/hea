"""Iterative sparse least squares — LSQR and LSMR.

Both solve ``min ||Ax - b||`` for a rectangular sparse ``A``, and both do it
**without forming ``AᵀA`` and without a factorization**. That is the whole
reason they are here beside a direct solver: the working set is ``A`` plus a
handful of vectors, where the direct route holds ``L``, and on a large system
``nnz(L)`` dominates everything else.

They also work with ``cond(A)`` where the normal equations work with
``cond(A)²``. Take that as a statement about the arithmetic, not as a promise
of a better answer: it bounds how much accuracy the *formulation* can cost,
and it says nothing about how close a run that stops on a tolerance actually
gets. Whether the bound is ever reached is a property of the matrix, and the
paragraph below on ``istop`` is the reason it usually is not.

The two are different in kind from :func:`cho_factor`, and callers should treat
them so:

* **They are iterative and stop on a tolerance.** Non-convergence is a normal
  outcome, not an error, and is reported through ``istop`` rather than raised.
  Nothing here raises :class:`CholmodError`; there is no factorization to fail.
* **The answer depends on ``atol``, ``btol``, ``conlim`` and the iteration
  limit.** A direct solve has no such knobs.
* **A successful ``istop`` is not a promise that the answer matches a direct
  solve.** The stopping tests are relative to estimates of ``‖A‖``, ``‖r‖`` and
  ``cond(A)`` that the iteration builds as it goes, so on an ill-conditioned
  system they can be satisfied while the residual is still orders of magnitude
  larger than the direct solve's, and the returned ``x`` a different vector.
  This is not a defect and it is not specific to this implementation — it is
  what a tolerance on an estimate means. Two consequences for a caller
  swapping a direct solve for one of these: **tighten ``atol``/``btol`` well
  below the default** on a badly conditioned problem, and **check the residual
  yourself** rather than reading ``istop`` as a verdict.

``istop`` is the stopping reason, on the scale both algorithms share:

===== ===============================================================
value meaning
===== ===============================================================
0     ``x = 0`` (or ``x = x0``) is already the solution
1     ``Ax - b`` is small enough, given ``atol`` and ``btol``
2     the least-squares solution is good enough, given ``atol``
3     the estimate of ``cond(Abar)`` has exceeded ``conlim``
4     ``Ax - b`` is small enough for this machine
5     the least-squares solution is good enough for this machine
6     ``cond(Abar)`` seems too large for this machine
7     the iteration limit was reached
===== ===============================================================

**Provenance.** These are mechanical ports of scipy 1.18.0's
``scipy.sparse.linalg.lsqr`` and ``lsmr``, which are themselves translations of
the Fortran and MATLAB originals: LSQR by C. C. Paige and M. A. Saunders
(TOMS 8(1) 43-71 and 8(2) 195-209, 1982), LSMR by D. C.-L. Fong and
M. A. Saunders. The signatures, the defaults and the return tuples are scipy's,
so a caller swaps the import and changes nothing else. Both originals are BSD
licensed.

Nothing here imports ``scipy.sparse.linalg``. ``A`` may be a scipy sparse
matrix, a dense array, or any object exposing ``shape`` together with
``matvec``/``rmatvec``.
"""

from __future__ import annotations

from math import sqrt

import numpy as np

__all__ = ["lsmr", "lsqr"]

_EPS = np.finfo(np.float64).eps


class _Operator:
    """``A`` as the pair of products the bidiagonalization needs.

    Stands in for ``scipy.sparse.linalg.aslinearoperator`` so this module does
    not import the package it exists to replace. An object that already offers
    ``matvec``/``rmatvec`` is used as it is; anything else is driven through
    ``@`` and its transpose.
    """

    __slots__ = ("_A", "_duck", "dtype", "shape")

    def __init__(self, A):
        duck = hasattr(A, "matvec") and hasattr(A, "rmatvec")
        if not duck and not hasattr(A, "shape"):
            A = np.asarray(A)
        shape = tuple(A.shape)
        if len(shape) != 2:
            raise ValueError(
                f"{len(shape)}-dimensional `A` is unsupported, expected 2-D."
            )
        self._A = A
        self._duck = duck
        self.shape = shape
        # `np.result_type` reads this, which is how the working precision is
        # derived from the operator rather than from a probe product.
        self.dtype = np.dtype(getattr(A, "dtype", np.float64))

    @property
    def ndim(self) -> int:
        return 2

    def matvec(self, x) -> np.ndarray:
        y = self._A.matvec(x) if self._duck else self._A @ x
        return np.asarray(y).reshape(self.shape[0])

    def rmatvec(self, x) -> np.ndarray:
        y = self._A.rmatvec(x) if self._duck else self._A.T @ x
        return np.asarray(y).reshape(self.shape[1])


def _sym_ortho(a, b):
    """Stable Givens rotation, as ``SymOrtho`` in the originals.

    Preferred to the direct formula because it removes the ``1/eps`` that
    appears when one of the two is far smaller than the other.
    """
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


_MSG = (
    "The exact solution is  x = 0                              ",
    "Ax - b is small enough, given atol, btol                  ",
    "The least-squares solution is good enough, given atol     ",
    "The estimate of cond(Abar) has exceeded conlim            ",
    "Ax - b is small enough for this machine                   ",
    "The least-squares solution is good enough for this machine",
    "Cond(Abar) seems to be too large for this machine         ",
    "The iteration limit has been reached                      ",
)


def lsqr(
    A,
    b,
    damp=0.0,
    atol=1e-6,
    btol=1e-6,
    conlim=1e8,
    iter_lim=None,
    show=False,
    calc_var=False,
    x0=None,
):
    """Least-squares solution of ``Ax = b`` by the LSQR bidiagonalization.

    Solves ``Ax = b``, or ``min ||Ax - b||²``, or with ``damp`` the ridge
    problem ``min ||Ax - b||² + damp² ||x - x0||²``. ``A`` may be square or
    rectangular and may have any rank.

    Parameters
    ----------
    A : array, sparse matrix, or operator, shape (m, n)
    b : array, shape (m,)
    damp : float
        Damping coefficient; ``0`` for plain least squares.
    atol, btol : float
        Stopping tolerances on the least-squares residual and on ``Ax - b``.
        The iteration stops when it can no longer distinguish the answer from
        one whose data has relative error at that level, so there is nothing to
        gain from values below machine precision.
    conlim : float
        Stop if the estimated ``cond(Abar)`` exceeds this.
    iter_lim : int, optional
        Defaults to ``2 * n``.
    show : bool
        Print a per-iteration table.
    calc_var : bool
        Also return an estimate of ``diag(inv(AᵀA))``.
    x0 : array, optional
        Starting guess, in which case ``damp`` penalizes ``x - x0``.

    Returns
    -------
    x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
        See the module docstring for ``istop``. Reaching ``iter_lim`` gives
        ``istop = 7`` and is not an error.
    """
    A = _Operator(A)
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    m, n = A.shape
    if iter_lim is None:
        iter_lim = 2 * n
    var = np.zeros(n)

    if show:
        print(" ")
        print("LSQR            Least-squares solution of  Ax = b")
        print(f"The matrix A has {m} rows and {n} columns")
        print(f"damp = {damp:20.14e}   calc_var = {calc_var:8g}")
        print(f"atol = {atol:8.2e}                 conlim = {conlim:8.2e}")
        print(f"btol = {btol:8.2e}               iter_lim = {iter_lim:8g}")

    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    # The first vectors of the bidiagonalization, satisfying
    # beta*u = b - A@x and alfa*v = A'@u.
    u = b
    bnorm = np.linalg.norm(b)

    if x0 is None:
        x = np.zeros(n)
        beta = bnorm.copy()
    else:
        x = np.asarray(x0)
        u = u - A.matvec(x)
        beta = np.linalg.norm(u)

    if beta > 0:
        u = (1 / beta) * u
        v = A.rmatvec(u)
        alfa = np.linalg.norm(v)
    else:
        v = x.copy()
        alfa = 0

    if alfa > 0:
        v = (1 / alfa) * v
    w = v.copy()

    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Ordered before the loop rather than after it, so arnorm == 0 returns
    # rather than dividing by it.
    arnorm = alfa * beta
    if arnorm == 0:
        if show:
            print(_MSG[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    head1 = "   Itn      x[0]       r1norm     r2norm "
    head2 = " Compatible    LS      Norm A   Cond A"

    if show:
        print(" ")
        print(head1, head2)
        test1 = 1
        test2 = alfa / beta
        str1 = f"{itn:6g} {x[0]:12.5e}"
        str2 = f" {r1norm:10.3e} {r2norm:10.3e}"
        str3 = f"  {test1:8.1e} {test2:8.1e}"
        print(str1, str2, str3)

    while itn < iter_lim:
        itn = itn + 1
        # The next step of the bidiagonalization, giving the next beta, u,
        # alfa, v, which satisfy
        #     beta*u  =  A@v   -  alfa*u,
        #     alfa*v  =  A'@u  -  beta*v.
        u = A.matvec(v) - alfa * u
        beta = np.linalg.norm(u)

        if beta > 0:
            u = (1 / beta) * u
            anorm = sqrt(anorm**2 + alfa**2 + beta**2 + dampsq)
            v = A.rmatvec(u) - beta * v
            alfa = np.linalg.norm(v)
            if alfa > 0:
                v = (1 / alfa) * v

        # A plane rotation eliminates the damping parameter, altering the
        # diagonal (rhobar) of the lower-bidiagonal matrix.
        if damp > 0:
            rhobar1 = sqrt(rhobar**2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            # cs1 = 1 and sn1 = 0
            rhobar1 = rhobar
            psi = 0.0

        # A second rotation eliminates the subdiagonal element (beta), taking
        # the lower-bidiagonal matrix to an upper-bidiagonal one.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w

        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + np.linalg.norm(dk) ** 2

        if calc_var:
            var = var + dk**2

        # A rotation on the right eliminates the super-diagonal element
        # (theta), and the result estimates norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar**2)
        gamma = sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Estimate the condition of Abar and the norms of rbar and Abar'rbar.
        acond = anorm * sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # r1norm = ||b - Ax||, r2norm = sqrt(r1norm² + damp²||x - x0||²), so
        # r1norm comes back from r2norm by subtraction. It cancels, but not
        # enough to matter here.
        if damp > 0:
            r1sq = rnorm**2 - dampsq * xxnorm
            r1norm = sqrt(abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm
        r2norm = rnorm

        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + _EPS)
        test3 = 1 / (acond + _EPS)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # The tests in this block guard against extremely small atol, btol or
        # ctol — a caller may have set any of them to 0 — and are equivalent to
        # the tests below run at atol = btol = eps, conlim = 1/eps.
        if itn >= iter_lim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # The tolerances the caller did set.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if show:
            prnt = False
            if n <= 40:
                prnt = True
            if itn <= 10:
                prnt = True
            if itn >= iter_lim - 10:
                prnt = True
            if test3 <= 2 * ctol:
                prnt = True
            if test2 <= 10 * atol:
                prnt = True
            if test1 <= 10 * rtol:
                prnt = True
            if istop != 0:
                prnt = True

            if prnt:
                str1 = f"{itn:6g} {x[0]:12.5e}"
                str2 = f" {r1norm:10.3e} {r2norm:10.3e}"
                str3 = f"  {test1:8.1e} {test2:8.1e}"
                str4 = f" {anorm:8.1e} {acond:8.1e}"
                print(str1, str2, str3, str4)

        if istop != 0:
            break

    if show:
        print(" ")
        print("LSQR finished")
        print(_MSG[istop])
        print(" ")
        str1 = f"istop ={istop:8g}   r1norm ={r1norm:8.1e}"
        str2 = f"anorm ={anorm:8.1e}   arnorm ={arnorm:8.1e}"
        str3 = f"itn   ={itn:8g}   r2norm ={r2norm:8.1e}"
        str4 = f"acond ={acond:8.1e}   xnorm  ={xnorm:8.1e}"
        print(str1 + "   " + str2)
        print(str3 + "   " + str4)
        print(" ")

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var


def lsmr(
    A,
    b,
    damp=0.0,
    atol=1e-6,
    btol=1e-6,
    conlim=1e8,
    maxiter=None,
    show=False,
    x0=None,
):
    """Least-squares solution of ``Ax = b`` by LSMR.

    Solves the same problems :func:`lsqr` does — ``Ax = b``,
    ``min ||Ax - b||``, and the damped variant — by the same Golub-Kahan
    bidiagonalization, but minimising ``||Aᵀ(Ax - b)||`` monotonically instead
    of ``||Ax - b||``. Both norms decrease under either method; which one is
    monotone decides which stopping test is reliable, so LSMR is the steadier
    choice when the system is inconsistent and the answer is a least-squares
    one rather than a solution.

    Parameters
    ----------
    A : array, sparse matrix, or operator, shape (m, n)
    b : array, shape (m,)
    damp : float
        Damping coefficient; ``0`` for plain least squares.
    atol, btol : float
        Stopping tolerances, as in :func:`lsqr`.
    conlim : float
        Stop if the estimated ``cond(A)`` exceeds this. The default is lower
        than :func:`lsqr`'s effective one because LSMR's estimate is the
        cheaper, cruder of the two.
    maxiter : int, optional
        Defaults to ``min(m, n)``. Larger is worthwhile for a matrix that is
        ill-conditioned rather than merely large.
    show : bool
        Print a per-iteration table.
    x0 : array, optional
        Starting guess.

    Returns
    -------
    x, istop, itn, normr, normar, norma, conda, normx
        See the module docstring for ``istop``. Reaching ``maxiter`` gives
        ``istop = 7`` and is not an error.
    """
    A = _Operator(A)
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    hdg1 = "   itn      x(1)       norm r    norm Ar"
    hdg2 = " compatible   LS      norm A   cond A"
    pfreq = 20  # print frequency (for repeating the heading)
    pcount = 0  # print counter

    m, n = A.shape

    minDim = min([m, n])

    if maxiter is None:
        maxiter = minDim

    if x0 is None:
        dtype = np.result_type(A, b, float)
    else:
        dtype = np.result_type(A, b, x0, float)

    if show:
        print(" ")
        print("LSMR            Least-squares solution of  Ax = b\n")
        print(f"The matrix A has {m} rows and {n} columns")
        print(f"damp = {damp:20.14e}\n")
        print(f"atol = {atol:8.2e}                 conlim = {conlim:8.2e}\n")
        print(f"btol = {btol:8.2e}             maxiter = {maxiter:8g}\n")

    u = b
    normb = np.linalg.norm(b)
    if x0 is None:
        x = np.zeros(n, dtype)
        beta = normb.copy()
    else:
        x = np.atleast_1d(x0.copy())
        u = u - A.matvec(x)
        beta = np.linalg.norm(u)

    if beta > 0:
        u = (1 / beta) * u
        v = A.rmatvec(u)
        alpha = np.linalg.norm(v)
    else:
        v = np.zeros(n, dtype)
        alpha = 0

    if alpha > 0:
        v = (1 / alpha) * v

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = np.zeros(n, dtype)

    # Variables for the estimate of ||r||.
    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Variables for the estimates of ||A|| and cond(A).
    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e100
    normA = sqrt(normA2)
    condA = 1
    normx = 0

    str1 = ""
    str2 = ""
    str3 = ""
    str4 = ""

    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Ordered before the loop rather than after it, so normar == 0 returns
    # rather than dividing by it.
    normar = alpha * beta
    if normar == 0:
        if show:
            print(_MSG[0])
        return x, istop, itn, normr, normar, normA, condA, normx

    if normb == 0:
        x[()] = 0
        return x, istop, itn, normr, normar, normA, condA, normx

    if show:
        print(" ")
        print(hdg1, hdg2)
        test1 = 1
        test2 = alpha / beta
        str1 = f"{itn:6g} {x[0]:12.5e}"
        str2 = f" {normr:10.3e} {normar:10.3e}"
        str3 = f"  {test1:8.1e} {test2:8.1e}"
        print(f"{str1}{str2}{str3}")

    while itn < maxiter:
        itn = itn + 1

        # The next step of the bidiagonalization, giving the next beta, u,
        # alpha, v, which satisfy
        #      beta*u  =  A@v   -  alpha*u,
        #     alpha*v  =  A'@u  -  beta*v.
        u *= -alpha
        u += A.matvec(v)
        beta = np.linalg.norm(u)

        if beta > 0:
            u *= 1 / beta
            v *= -beta
            v += A.rmatvec(u)
            alpha = np.linalg.norm(v)
            if alpha > 0:
                v *= 1 / alpha

        # Here beta = beta_{k+1} and alpha = alpha_{k+1}.

        # Rotation Qhat_{k,2k+1}.
        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Rotation Q_i takes B_i to R_i.
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        # Rotation Qbar_i takes R_i^T to R_i^bar.
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar

        # Update h, hbar, x.
        hbar *= -(thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += (zeta / (rho * rhobar)) * hbar
        h *= -(thetanew / rho)
        h += v

        # Estimate ||r||, by applying rotation Qhat_{k,2k+1} ...
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # ... then Q_{k,k+1} ...
        betahat = c * betaacute
        betadd = -s * betaacute

        # ... then Qtilde_{k-1}, where betad is betad_{k-1} on entry.
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat

        # Now betad is betad_k and rhodold is rhod_k.
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = sqrt(d + (betad - taud) ** 2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        normar = abs(zetabar)
        normx = np.linalg.norm(x)

        test1 = normr / normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = np.inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The tests in this block guard against extremely small atol, btol or
        # ctol — a caller may have set any of them to 0 — and are equivalent to
        # the tests below run at atol = btol = eps, conlim = 1/eps.
        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # The tolerances the caller did set.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if show and (
            (n <= 40)
            or (itn <= 10)
            or (itn >= maxiter - 10)
            or (itn % 10 == 0)
            or (test3 <= 1.1 * ctol)
            or (test2 <= 1.1 * atol)
            or (test1 <= 1.1 * rtol)
            or (istop != 0)
        ):
            if pcount >= pfreq:
                pcount = 0
                print(" ")
                print(hdg1, hdg2)
            pcount = pcount + 1
            str1 = f"{itn:6g} {x[0]:12.5e}"
            str2 = f" {normr:10.3e} {normar:10.3e}"
            str3 = f"  {test1:8.1e} {test2:8.1e}"
            str4 = f" {normA:8.1e} {condA:8.1e}"
            print(f"{str1}{str2}{str3}{str4}")

        if istop > 0:
            break

    if show:
        print(" ")
        print("LSMR finished")
        print(_MSG[istop])
        print(f"istop ={istop:8g}    normr ={normr:8.1e}")
        print(f"    normA ={normA:8.1e}    normAr ={normar:8.1e}")
        print(f"itn   ={itn:8g}    condA ={condA:8.1e}")
        print(f"    normx ={normx:8.1e}")
        print(str1, str2)
        print(str3, str4)

    return x, istop, itn, normr, normar, normA, condA, normx
