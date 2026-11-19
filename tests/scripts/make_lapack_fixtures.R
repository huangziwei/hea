## R's own LAPACK results on fixed inputs, for tests/test_lapack.py: base R's
## eigen(symmetric = TRUE) (LAPACK dsyevr) and mgcv's slanczos (Lanczos with
## LAPACK dstedc on its tridiagonal). Run from the repository root:
##
##     Rscript tests/scripts/make_lapack_fixtures.R
##
## The inputs are rebuilt in Python from the same set.seed()/runif() draws
## (hea.R.rng reproduces R's generator) or from integer arithmetic, so only
## the outputs are stored, as raw doubles (writeBin) to keep every bit.
suppressPackageStartupMessages(library(mgcv))
out <- "tests/fixtures/lapack"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
wb <- function(x, f) writeBin(as.double(x), file.path(out, f))
unif_sym <- function(seed, n) {
  set.seed(seed)
  U <- matrix(runif(n * n), n, n)
  U + t(U)
}
cases <- list(
  rand6 = unif_sym(1, 6),
  rand40 = unif_sym(2, 40),
  rand140 = unif_sym(3, 140),
  diffpen50 = crossprod(diff(diag(50), differences = 2)),
  clus36 = {
    set.seed(4)
    kronecker(diag(12), matrix(1, 3, 3)) + diag(runif(36)) * 1e-10
  },
  asym45 = {
    set.seed(9)
    matrix(runif(45 * 45), 45, 45)
  }
)
for (nm in names(cases)) {
  e <- eigen(cases[[nm]], symmetric = TRUE)
  wb(e$values, paste0("eigen_", nm, "_values.bin"))
  wb(e$vectors, paste0("eigen_", nm, "_vectors.bin"))
  v <- eigen(cases[[nm]], symmetric = TRUE, only.values = TRUE)$values
  wb(v, paste0("eigen_", nm, "_onlyvalues.bin"))
}
for (s in list(c(5, 120, 10), c(6, 300, 20))) {
  er <- slanczos(unif_sym(s[1], s[2]), s[3], -1)
  tag <- paste0("slanczos_", s[2], "_", s[3])
  wb(er$values, paste0(tag, "_values.bin"))
  wb(er$vectors, paste0(tag, "_vectors.bin"))
}

## qr(LAPACK = TRUE) (dgeqp3) with qr.qty / qr.qy / qr.R (dormqr), the
## default LINPACK qr with qr.coef / qr.qy on a rank-deficient matrix, and
## solve() (dgesv).
set.seed(41)
X <- matrix(runif(150 * 140), 150, 140)
Y <- matrix(runif(150 * 3), 150, 3)
q <- qr(X, LAPACK = TRUE)
wb(q$qr, "qrl_qr.bin"); wb(q$qraux, "qrl_qraux.bin"); wb(q$pivot, "qrl_pivot.bin")
wb(qr.qty(q, Y), "qrl_qty.bin"); wb(qr.qy(q, Y), "qrl_qy.bin"); wb(qr.R(q), "qrl_R.bin")
set.seed(42)
Z <- matrix(runif(40 * 9), 40, 9)
Z[, 9] <- Z[, 1] + Z[, 2]
W <- matrix(runif(40 * 2), 40, 2)
l <- qr(Z)
wb(l$qr, "qr_qr.bin"); wb(l$qraux, "qr_qraux.bin"); wb(l$pivot, "qr_pivot.bin")
wb(l$rank, "qr_rank.bin"); wb(qr.coef(l, W), "qr_coef.bin"); wb(qr.qy(l, W), "qr_qy.bin")
set.seed(43)
A <- matrix(runif(70 * 70), 70, 70)
B <- matrix(runif(70 * 2), 70, 2)
wb(solve(A, B), "solve_ab.bin"); wb(solve(A), "solve_inv.bin")

## lm.fit's Cdqrls (LINPACK dqrls): a short design (R keeps vectors of
## length 2 to 16 off the 16-byte boundary) and a rank-deficient one.
set.seed(44)
Xs <- matrix(runif(12 * 4), 12, 4); ys <- runif(12)
Xd <- matrix(runif(60 * 6), 60, 6); Xd[, 6] <- Xd[, 1] - Xd[, 2]; yd <- runif(60)
for (nm in c("s", "d")) {
  X <- get(paste0("X", nm)); y <- get(paste0("y", nm))
  z <- .Call(stats:::C_Cdqrls, X, y, 1e-7, FALSE)
  wb(c(z$coefficients, z$residuals, z$effects, z$qraux, z$qr), paste0("cdqrls_", nm, ".bin"))
}

## svd() (La.svd's dgesdd: jobu 'S' by default, 'N' for nu = nv = 0, 'A' for
## nu = nrow) on tall, wide and square matrices, past the 25-row
## divide-and-conquer threshold and the 11/6 QR-first ratio.
for (s in list(c(51, 12, 8), c(52, 40, 40), c(53, 30, 70), c(54, 200, 30),
               c(55, 7, 5), c(56, 3, 3), c(57, 140, 140))) {
  set.seed(s[1])
  X <- matrix(runif(s[2] * s[3]), s[2], s[3])
  tag <- paste0("svd_", s[2], "x", s[3])
  sv <- svd(X)
  wb(c(sv$d, sv$u, sv$v), paste0(tag, "_S.bin"))
  wb(svd(X, 0, 0)$d, paste0(tag, "_N.bin"))
  sa <- svd(X, nu = nrow(X), nv = ncol(X))
  wb(c(sa$d, sa$u, sa$v), paste0(tag, "_A.bin"))
}
