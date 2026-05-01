## Generate the chicago oracle for hea's bam(discrete=TRUE) parity test.
##
## Two models:
##   1. chicago_simple — multi-smooth s() Poisson, no matrix-arg.
##   2. chicago_lag    — matrix-arg te() with distributed-lag pm10.
##
## Saves coef, fitted, sp, edf to tests/fixtures/chicago/<model>/.
suppressMessages({
  library(mgcv)
  library(gamair)
})

data(chicago)
ap <- na.omit(chicago[, c("death", "pm10median", "o3median",
                          "so2median", "time", "tmpd")])

cat("=== chicago after na.omit:", nrow(ap), "rows ===\n")
write.csv(ap, "/Users/ziweih/Works/hea/tests/fixtures/chicago/data.csv",
          row.names = FALSE)

dump_one <- function(name, model_fn) {
  cat("\n=== fitting", name, "===\n")
  out_dir <- paste0("/Users/ziweih/Works/hea/tests/fixtures/chicago/", name)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  m <- model_fn()
  cat("  iterations:", m$iter, "\n")
  cat("  sp:", m$sp, "\n")
  cat("  fREML:", m$gcv.ubre, "\n")
  cat("  edf sum:", sum(m$edf), "\n")
  cat("  coef[1:3]:", coef(m)[1:3], "\n")
  cat("  fitted[1:3]:", fitted(m)[1:3], "\n")
  write.table(coef(m), file.path(out_dir, "coef.csv"),
              row.names = FALSE, col.names = FALSE)
  write.table(fitted(m), file.path(out_dir, "fitted.csv"),
              row.names = FALSE, col.names = FALSE)
  write.table(m$sp, file.path(out_dir, "sp.csv"),
              row.names = FALSE, col.names = FALSE)
  write.table(m$edf, file.path(out_dir, "edf.csv"),
              row.names = FALSE, col.names = FALSE)
  saveRDS(m$gcv.ubre, file.path(out_dir, "freml.rds"))
}

## Model 1: simple multi-smooth s() chicago
dump_one("simple", function() {
  bam(death ~ s(time, k = 200) + s(pm10median) + s(o3median) +
                s(so2median) + s(tmpd),
      family = poisson, data = ap, discrete = TRUE)
})

## Model 2: distributed-lag matrix-arg te()
##   Build lag matrices for pm10median (6 lags) and a sibling lag-index
##   matrix (constant across rows). This is the canonical
##   ``?bam.examples`` style lag fit; mgcv's matrix-arg discrete path
##   is exactly what hea's Phase 1 fix targets.
lagard <- function(x, n.lag = 6) {
  n <- length(x); X <- matrix(NA, n, n.lag)
  for (i in 1:n.lag) X[i:n, i] <- x[i:n - i + 1]
  X
}

dump_one("lag", function() {
  ap2 <- ap
  pm10_lag <- lagard(ap2$pm10median, n.lag = 6)
  lag_idx <- matrix(0:5, nrow = nrow(ap2), ncol = 6, byrow = TRUE)
  ## drop NA rows from the lag-induced incomplete head
  good <- complete.cases(pm10_lag)
  ap2 <- ap2[good, ]
  pm10_lag <- pm10_lag[good, ]
  lag_idx <- lag_idx[good, ]
  cat("  lag-trimmed rows:", nrow(ap2), "\n")
  ap2$pm10_lag <- pm10_lag
  ap2$lag_idx <- lag_idx
  bam(death ~ s(time, k = 200) +
              te(pm10_lag, lag_idx, k = c(10, 5)) +
              s(tmpd),
      family = poisson, data = ap2, discrete = TRUE)
})

cat("\n=== Done — fixtures in tests/fixtures/chicago/ ===\n")
