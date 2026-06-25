## RF (signal-regression) oracle for hea's bam matrix-argument by= parity
## test — the RF1 blocker fix (gam-plan ★S1d / bam-plan RF1).
##
## Model: Poisson `y ~ te(Lag, Xc, by=Stim, k=c(4,3))` and a 1-D
## `y ~ s(Lag, by=Stim, k=8)`, both with MATRIX arguments (mgcv summation
## convention / distributed-lag signal regression). `Stim` is the matrix
## by= weight; `Lag`/`Xc` are the coordinate matrices. This is the native
## scalable RF path: bam, not gam (gam has no discrete path).
##
## Two mgcv references are dumped per model:
##   * bamT = bam(discrete=TRUE) — DISCRETISES (bins) the by-variable
##     (bam.r:2470-2482: the by is the first tensor marginal,
##     matrix(by.var, nr, 1), indexed by dk$k; weight by.var[dk$k]). hea's
##     bam(discrete=TRUE) bins the by the SAME way (RF1a, 2026-06-17), so
##     hea-discrete pins to bamT TIGHTLY.
##   * bamF = bam(discrete=FALSE) — the EXACT (un-binned) by= + full REML.
##     For a continuous Stim the binning is lossy, so bamF differs from bamT
##     (and from hea-discrete) by ~1e-3; bamF is the home of hea's
##     bam(discrete=FALSE). Pinned loosely to document that gap.
##
## Data is dumped so python reproduces the identical fit without R's RNG.
suppressMessages(library(mgcv))

dump_model <- function(tag, f, dat) {
  bF <- bam(f, family = poisson, data = dat, discrete = FALSE)
  bT <- bam(f, family = poisson, data = dat, discrete = TRUE)
  cat(sprintf("%-4s bamF: edf=%.6f sp=(%s)\n", tag, sum(bF$edf),
              paste(sprintf("%.5g", bF$sp), collapse = ", ")))
  cat(sprintf("%-4s bamT: edf=%.6f sp=(%s)\n", tag, sum(bT$edf),
              paste(sprintf("%.5g", bT$sp), collapse = ", ")))
  d <- file.path("/Users/ziweih/Works/hea/tests/fixtures/bam_rf_by", tag)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  write.table(fitted(bF), file.path(d, "bamF_fitted.csv"),
              row.names = FALSE, col.names = FALSE)
  write.table(bF$sp,  file.path(d, "bamF_sp.csv"),  row.names = FALSE, col.names = FALSE)
  write.table(sum(bF$edf), file.path(d, "bamF_edfsum.csv"), row.names = FALSE, col.names = FALSE)
  write.table(fitted(bT), file.path(d, "bamT_fitted.csv"),
              row.names = FALSE, col.names = FALSE)
  write.table(sum(bT$edf), file.path(d, "bamT_edfsum.csv"), row.names = FALSE, col.names = FALSE)
}

out_dir <- "/Users/ziweih/Works/hea/tests/fixtures/bam_rf_by"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## --- 2-D spatiotemporal RF: te(Lag, Xc, by=Stim) ------------------------
set.seed(1)
n <- 400; nlag <- 8; nx <- 6; m <- nlag * nx
lag_grid <- rep(0:(nlag - 1), times = nx)
x_grid   <- rep(seq(0, 1, length.out = nx), each = nlag)
Lag  <- matrix(rep(lag_grid, each = n), n, m)
Xc   <- matrix(rep(x_grid,   each = n), n, m)
Stim <- matrix(rnorm(n * m), n, m)
w    <- exp(-((lag_grid - 2)^2 * 0.5 + (x_grid * 4 - 2)^2))
eta  <- as.numeric(Stim %*% w) * 0.3 - 1
y    <- rpois(n, exp(eta))
write.csv(data.frame(y = y),  file.path(out_dir, "te_y.csv"),    row.names = FALSE)
write.csv(as.data.frame(Lag), file.path(out_dir, "te_Lag.csv"),  row.names = FALSE)
write.csv(as.data.frame(Xc),  file.path(out_dir, "te_Xc.csv"),   row.names = FALSE)
write.csv(as.data.frame(Stim),file.path(out_dir, "te_Stim.csv"), row.names = FALSE)
dump_model("te", y ~ te(Lag, Xc, by = Stim, k = c(4, 3)),
           list(y = y, Lag = Lag, Xc = Xc, Stim = Stim))

## RF4: extended-family (Tweedie, fixed p) + matrix-by on the SAME te data.
## Confirms the extended-family fREML cadence (B2) composes with the matrix-arg
## summation design. discrete=FALSE (exact-by); hea pins to this. NOTE: the te
## data is near-ridge (one margin sp ~1e8), so the Tweedie scale estimate is
## sensitive to tiny data perturbations — re-read the dumped CSVs so the oracle
## fits the EXACT bytes hea sees (an in-memory fit drifts ~7e-4 in edf vs the
## CSV round-trip; Poisson is insensitive to it but Tweedie is not).
tw_y  <- as.numeric(read.csv(file.path(out_dir, "te_y.csv"))[[1]])
tw_L  <- as.matrix(read.csv(file.path(out_dir, "te_Lag.csv")))
tw_X  <- as.matrix(read.csv(file.path(out_dir, "te_Xc.csv")))
tw_S  <- as.matrix(read.csv(file.path(out_dir, "te_Stim.csv")))
bTw <- bam(y ~ te(Lag, Xc, by = Stim, k = c(4, 3)), family = Tweedie(p = 1.5),
           data = list(y = tw_y, Lag = tw_L, Xc = tw_X, Stim = tw_S),
           discrete = FALSE)
cat(sprintf("te   Tweedie(1.5) bamF: edf=%.6f scale=%.5f\n", sum(bTw$edf), bTw$scale))
write.table(fitted(bTw), file.path(out_dir, "te", "bamF_tw_fitted.csv"),
            row.names = FALSE, col.names = FALSE)
write.table(sum(bTw$edf), file.path(out_dir, "te", "bamF_tw_edfsum.csv"),
            row.names = FALSE, col.names = FALSE)

## --- 1-D temporal RF: s(Lag, by=Stim) -----------------------------------
set.seed(2)
n <- 400; m <- 30
lag_grid <- 0:(m - 1)
Lag  <- matrix(rep(lag_grid, each = n), n, m)
Stim <- matrix(rnorm(n * m), n, m)
w    <- exp(-((lag_grid - 5)^2 * 0.1))
eta  <- as.numeric(Stim %*% w) * 0.3 - 1
y    <- rpois(n, exp(eta))
write.csv(data.frame(y = y),   file.path(out_dir, "s_y.csv"),    row.names = FALSE)
write.csv(as.data.frame(Lag),  file.path(out_dir, "s_Lag.csv"),  row.names = FALSE)
write.csv(as.data.frame(Stim), file.path(out_dir, "s_Stim.csv"), row.names = FALSE)
dump_model("s", y ~ s(Lag, by = Stim, k = 8),
           list(y = y, Lag = Lag, Stim = Stim))

## --- 2-D NEAR-RIDGE RF: te(Lag, Xc, by=Stim) on an asymmetric surface ----
## RF1b regression guard: the true RF varies strongly in Lag but is ~flat in
## Xc, so the fREML sp-search pushes the Xc margin onto a near-linear ridge
## (sp ~ 5e3) — the regime where hea's OLD exact-by design landed in a worse
## REML basin than mgcv (edf 6.81 vs 10.65). After RF1a (bin the by like
## mgcv) hea bam(discrete=TRUE) tracks mgcv bamT into the SAME basin. This
## case is well-determined (sp ~5472 vs ~30, both modes agree to 1e-4), so it
## is a stable pin — not flat-optimum indeterminacy.
set.seed(11)
n <- 400; nlag <- 8; nx <- 5; m <- nlag * nx
lag_grid <- rep(0:(nlag - 1), times = nx)
x_grid   <- rep(seq(0, 1, length.out = nx), each = nlag)
Lag  <- matrix(rep(lag_grid, each = n), n, m)
Xc   <- matrix(rep(x_grid,   each = n), n, m)
Stim <- matrix(rnorm(n * m), n, m)
w    <- exp(-((lag_grid - 2)^2 * 0.5)) * (1 + 0.05 * (x_grid * 4 - 2))
eta  <- as.numeric(Stim %*% w) * 0.3 - 1
y    <- rpois(n, exp(eta))
write.csv(data.frame(y = y),  file.path(out_dir, "te_ridge_y.csv"),    row.names = FALSE)
write.csv(as.data.frame(Lag), file.path(out_dir, "te_ridge_Lag.csv"),  row.names = FALSE)
write.csv(as.data.frame(Xc),  file.path(out_dir, "te_ridge_Xc.csv"),   row.names = FALSE)
write.csv(as.data.frame(Stim),file.path(out_dir, "te_ridge_Stim.csv"), row.names = FALSE)
dump_model("te_ridge", y ~ te(Lag, Xc, by = Stim, k = c(4, 3)),
           list(y = y, Lag = Lag, Xc = Xc, Stim = Stim))

cat("\nwrote oracle to", out_dir, "\n")
