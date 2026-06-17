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
##   * bamF = bam(discrete=FALSE) — fits with the EXACT (un-discretised)
##     by= and full REML. hea's bam(discrete=TRUE) uses the exact by= too
##     (it does not yet discretise the by-variable — see bam-plan RF1
##     follow-up), so hea-discrete must pin to bamF TIGHTLY.
##   * bamT = bam(discrete=TRUE) — additionally DISCRETISES the by-variable
##     (bam.r:2469-2483 represents it as the first tensor marginal, indexed
##     by dk$k). That binning is lossy for a continuous Stim, so bamT
##     differs from bamF (and from hea) by ~1e-3. Pinned loosely to
##     document the residual by-discretisation gap.
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

cat("\nwrote oracle to", out_dir, "\n")
