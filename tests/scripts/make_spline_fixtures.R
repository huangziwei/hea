## R's B-spline results on fixed inputs, for tests/test_rsplines.py:
## splines::splineDesign (values, derivatives, outer.ok, repeated knots),
## splines::bs()/ns() with explicit knots and x beyond the boundary knots,
## and mgcv's smoothCon/PredictMat for the B-spline classes (ps, cp, ad).
## Run from the repository root:
##
##     Rscript tests/scripts/make_spline_fixtures.R
##
## Inputs are runif() draws (rebuilt in Python by hea.R.rng) transformed by
## plain R arithmetic, and dyadic constants, so both sides hold the same
## bits; outputs are raw doubles (writeBin).
suppressPackageStartupMessages({library(splines); library(mgcv)})
out <- "tests/fixtures/splines"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
wb <- function(x, f) writeBin(as.double(x), file.path(out, f))

set.seed(21)
knots <- sort(-2 + 5 * runif(12))
x <- knots[4] + (knots[9] - knots[4]) * runif(60)
wb(splineDesign(knots, x, 4), "sd_values.bin")
wb(splineDesign(knots, x, 4, derivs = rep(0:3, 15)), "sd_derivs.bin")
x2 <- -3 + 7 * runif(40)
wb(splineDesign(knots, x2, 3, outer.ok = TRUE), "sd_outer.bin")
k4 <- c(rep(0, 4), sort(runif(5)), rep(1, 4))
x4 <- c(runif(30), 0, 1, k4[6])
wb(splineDesign(k4, x4, 4), "sd_repeated.bin")
wb(splineDesign(k4, x4, 4, derivs = 2), "sd_repeated_d2.bin")

set.seed(22)
x5 <- -2 + 7 * runif(120)
kn <- c(-0.5, 0.75, 1.875, 3.25)
wb(bs(x5, knots = kn, Boundary.knots = c(-1, 4), warn.outside = FALSE), "bs3.bin")
wb(bs(x5, knots = kn, degree = 2, Boundary.knots = c(-1, 4), intercept = TRUE,
      warn.outside = FALSE), "bs2_int.bin")
wb(ns(x5, knots = kn, Boundary.knots = c(-1, 4)), "ns.bin")
wb(ns(x5, knots = kn, Boundary.knots = c(-1, 4), intercept = TRUE), "ns_int.bin")

set.seed(23)
dat <- data.frame(x = 3 * runif(150), z = -1 + 2 * runif(150))
dat$y <- dat$x
nd <- data.frame(x = c(-0.375, 0, 1.25, 2.9375, 3.25, 3.625, 0.75),
                 z = c(0.125, -1.25, 0.5, 0.875, 1.375, -0.25, 0))
specs <- list(ps = s(x, bs = "ps"), cp = s(x, bs = "cp"), ad = s(x, bs = "ad", k = 20),
              ad2 = s(x, z, bs = "ad", k = 8))
for (nm in names(specs)) {
  sm <- smoothCon(specs[[nm]], data = dat, knots = NULL, absorb.cons = TRUE,
                  scale.penalty = TRUE)[[1]]
  wb(sm$X, paste0("sc_", nm, "_X.bin"))
  for (j in seq_along(sm$S)) wb(sm$S[[j]], sprintf("sc_%s_S%d.bin", nm, j))
  wb(PredictMat(sm, nd), paste0("sc_", nm, "_P.bin"))
}

## The bs smooth's derivative penalties (W1 quadrature, bandchol, crossprod)
## across penalty orders, without constraints or scaling.
set.seed(9)
dbs <- data.frame(x = runif(80))
bs_specs <- list(b32 = s(x, bs = "bs"), b5 = s(x, bs = "bs", k = 5),
                 b321 = s(x, bs = "bs", m = c(3, 2, 1)), b42 = s(x, bs = "bs", m = c(4, 2)),
                 b21 = s(x, bs = "bs", m = c(2, 1)), b33 = s(x, bs = "bs", m = c(3, 3)),
                 b11 = s(x, bs = "bs", m = c(1, 1)), b52 = s(x, bs = "bs", m = c(5, 2), k = 12))
for (nm in names(bs_specs)) {
  sm <- smoothCon(bs_specs[[nm]], data = dbs, absorb.cons = FALSE, scale.penalty = FALSE)[[1]]
  for (j in seq_along(sm$S)) wb(sm$S[[j]], sprintf("bspen_%s_S%d.bin", nm, j))
}

## bs()/ns() interior knots placed from df: quantile type 7 on the points
## inside Boundary.knots, with ties at a boundary shoved inside.
set.seed(31)
kx <- list(u = runif(97), r = round(runif(60) * 5) / 5, z = c(rep(0, 30), runif(40)),
           e = rexp(200))
for (nm in names(kx)) {
  wb(kx[[nm]], sprintf("dfk_%s_x.bin", nm))
  for (df in c(4, 5, 7, 12)) {
    wb(suppressWarnings(attr(bs(kx[[nm]], df = df), "knots")), sprintf("dfk_%s_bs%d.bin", nm, df))
    wb(suppressWarnings(attr(ns(kx[[nm]], df = df), "knots")), sprintf("dfk_%s_ns%d.bin", nm, df))
  }
}
