## End-to-end ``bam(family=scat(...), discrete=TRUE)`` oracle.
##
## Two fits — both Poisson-tailed-like data fit with the scaled-t family:
##
##   simple : y ~ s(x, k=10)                     — single smooth, no by
##   factor : y ~ g + s(x, by=g, k=10)           — factor-by smooth
##
## ``simple`` exercises the basic extended-family PIRLS path
## (Phase B/C/E), ``factor`` exercises the by=factor discrete-path fix
## (de-risk fix) under an extended family, plus the ``preinitialize``
## + ``estimate.theta`` interplay.
##
## Pinned outputs (under tests/fixtures/scat_bam/{simple,factor}/):
##   data.csv    — full input data frame
##   coef.csv    — coef(m)        (NOT pinned element-wise in tests; see below)
##   sp.csv      — m$sp           (smoothing parameters, one per slot)
##   theta.csv   — m$family$getTheta(TRUE)   (ν, σ on original scale)
##   fitted.csv  — fitted(m) on the response scale (μ̂)
##
## Tolerances on the python side (test_bam_scat.py):
##   force-θ-and-sp: fitted ≤ 1e-9 (gauge-invariant predictive equiv).
##   auto-fit:       fitted ≤ 1e-4, θ ≤ 1e-4, sp within ~3× (REML basin).
##
## Re-run after upgrading mgcv to refresh.

suppressMessages(library(mgcv))


.dump <- function(m, dir) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  writeLines(format(m$sp, digits = 15), file.path(dir, "sp.csv"))
  writeLines(format(m$family$getTheta(TRUE), digits = 15),
             file.path(dir, "theta.csv"))
  writeLines(format(fitted(m), digits = 15),
             file.path(dir, "fitted.csv"))
  write.csv(coef(m), file.path(dir, "coef.csv"), row.names = TRUE)
}


# ---------- simple ----------------------------------------------------------

set.seed(7)
n <- 250
x <- runif(n)
mu <- 2 * sin(2 * pi * x)
y <- mu + rt(n, df = 5) * 0.3
df1 <- data.frame(y = y, x = x)
m1 <- bam(y ~ s(x, k = 10), data = df1,
          family = scat(min.df = 5), discrete = TRUE)
write.csv(df1, "tests/fixtures/scat_bam/simple/data.csv", row.names = FALSE)
.dump(m1, "tests/fixtures/scat_bam/simple")
cat("simple: converged=", m1$converged, " iter=", m1$iter,
    " theta(ν,σ)=", format(m1$family$getTheta(TRUE), digits = 8), "\n")


# ---------- factor ----------------------------------------------------------

set.seed(11)
n <- 300
x <- runif(n)
g <- factor(rep(c("a", "b", "c"), length.out = n))
mu <- 2 * sin(2 * pi * x) + as.numeric(g) * 0.3
y <- mu + rt(n, df = 5) * 0.25
df2 <- data.frame(y = y, x = x, g = g)
m2 <- bam(y ~ g + s(x, by = g, k = 10), data = df2,
          family = scat(min.df = 5), discrete = TRUE)
write.csv(df2, "tests/fixtures/scat_bam/factor/data.csv", row.names = FALSE)
.dump(m2, "tests/fixtures/scat_bam/factor")
cat("factor: converged=", m2$converged, " iter=", m2$iter,
    " theta(ν,σ)=", format(m2$family$getTheta(TRUE), digits = 8), "\n")
