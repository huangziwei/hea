## prior-weights oracle for hea.models.bam(weights=) (plan item P2).
##
## hea-bam used to raise TypeError on weights= and hardcode self._wt = ones.
## Now weights thread through the chunked QR build (√w row scaling), the PIRLS
## Fisher weights (w·μ_η²/V), and every post-fit consumer (scale, leverage,
## residuals, null deviance, R²). These oracles pin hea-bam(weights=w) against
## mgcv-bam(weights=w) on a Gaussian and a Poisson fit.
##
## Dumps (under tests/fixtures/bam_weights/{gauss,pois}/):
##   data.csv   — input frame incl. a `w` weights column
##   coef.csv   — coef(m)
##   fitted.csv — fitted(m) on the response scale
##   meta.csv   — scale (sig2), edf_total, rank, dev.expl, plus the
##                intercept Estimate/SE from summary.gam's p.table

suppressMessages(library(mgcv))

.dump <- function(m, dir) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  write.csv(coef(m), file.path(dir, "coef.csv"), row.names = TRUE)
  writeLines(format(fitted(m), digits = 15), file.path(dir, "fitted.csv"))
  s <- summary(m)
  meta <- data.frame(
    scale     = s$scale,
    edf_total = sum(m$edf),
    rank      = m$rank,
    dev.expl  = s$dev.expl,
    int_est   = s$p.table["(Intercept)", "Estimate"],
    int_se    = s$p.table["(Intercept)", "Std. Error"]
  )
  write.csv(meta, file.path(dir, "meta.csv"), row.names = FALSE)
}

# ---------- gauss : weighted Gaussian-identity ------------------------------
set.seed(10)
n <- 150
x <- runif(n)
z <- runif(n)
w <- sample(1:5, n, replace = TRUE)               # analytic prior weights
y <- 0.3 + sin(2 * pi * x) + 0.5 * z + rnorm(n, sd = 0.5)
dg <- data.frame(y = y, x = x, z = z, w = w)
mg <- bam(y ~ z + s(x, k = 10), data = dg, weights = w)
write.csv(dg, "tests/fixtures/bam_weights/gauss/data.csv", row.names = FALSE)
.dump(mg, "tests/fixtures/bam_weights/gauss")
cat("gauss: scale=", format(summary(mg)$scale, digits = 8),
    " edf=", format(sum(mg$edf), digits = 8), "\n")

# ---------- pois : weighted Poisson-log -------------------------------------
set.seed(11)
n <- 150
x <- runif(n)
z <- runif(n)
w <- sample(1:4, n, replace = TRUE)
eta <- 0.4 + 0.8 * sin(2 * pi * x) + 0.5 * z
y <- rpois(n, exp(eta))
dp <- data.frame(y = y, x = x, z = z, w = w)
mp <- bam(y ~ z + s(x, k = 10), data = dp, family = poisson(), weights = w)
write.csv(dp, "tests/fixtures/bam_weights/pois/data.csv", row.names = FALSE)
.dump(mp, "tests/fixtures/bam_weights/pois")
cat("pois: scale=", format(summary(mp)$scale, digits = 8),
    " edf=", format(sum(mp$edf), digits = 8), "\n")
