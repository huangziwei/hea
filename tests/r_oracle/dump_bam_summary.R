## summary()/rank oracle for hea.models.bam (P1 + P5).
##
## The previous bam suite never called summary()/check()/anova(), so the
## inherited summary() machinery (which reads object$_se_report / object$rank)
## was never exercised on a bam. These oracles pin the numbers summary.gam
## prints — parametric table, smooth table, scale, R-sq, dev.expl, the fREML
## criterion value — and object$rank, across three fits:
##
##   gauss   : Gaussian-identity bam, full rank          (_post_fit_gaussian)
##   pois    : Poisson-log bam, scale known              (_post_fit_pirls)
##   rankdef : a rank-deficient design (te on too-few points) so the
##             `Rank: r/p` line and oo$rank.est < p are exercised.
##
## Dumps (under tests/fixtures/bam_summary/{gauss,pois,rankdef}/):
##   data.csv   — input frame
##   ptable.csv — summary(m)$p.table  (Estimate, Std. Error, t/z, p)
##   stable.csv — summary(m)$s.table  (edf, Ref.df, F or Chi.sq, p)
##   meta.csv   — rank, p, scale (sig2), r.sq, dev.expl, sp.criterion, n
##
## Re-run after upgrading mgcv to refresh.

suppressMessages(library(mgcv))

.dump <- function(m, dir) {
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  s <- summary(m)
  write.csv(s$p.table, file.path(dir, "ptable.csv"), row.names = TRUE)
  write.csv(s$s.table, file.path(dir, "stable.csv"), row.names = TRUE)
  meta <- data.frame(
    rank        = m$rank,
    p           = length(coef(m)),
    scale       = s$scale,
    r.sq        = s$r.sq,
    dev.expl    = s$dev.expl,
    sp.criterion= as.numeric(s$sp.criterion),
    n           = s$n
  )
  write.csv(meta, file.path(dir, "meta.csv"), row.names = FALSE)
}

# ---------- gauss : Gaussian-identity, full rank ----------------------------
set.seed(1)
n <- 200
x <- runif(n)
z <- runif(n)
y <- 0.3 + sin(2 * pi * x) + 0.5 * z + rnorm(n, sd = 0.4)
dg <- data.frame(y = y, x = x, z = z)
mg <- bam(y ~ z + s(x, k = 10), data = dg)
write.csv(dg, "tests/fixtures/bam_summary/gauss/data.csv", row.names = FALSE)
.dump(mg, "tests/fixtures/bam_summary/gauss")
cat("gauss: rank=", mg$rank, "/", length(coef(mg)),
    " scale=", format(summary(mg)$scale, digits = 10), "\n")

# ---------- pois : Poisson-log, scale known ---------------------------------
set.seed(2)
n <- 200
x <- runif(n)
z <- runif(n)
eta <- 0.4 + 0.8 * sin(2 * pi * x) + 0.5 * z
y <- rpois(n, exp(eta))
dp <- data.frame(y = y, x = x, z = z)
mp <- bam(y ~ z + s(x, k = 10), data = dp, family = poisson())
write.csv(dp, "tests/fixtures/bam_summary/pois/data.csv", row.names = FALSE)
.dump(mp, "tests/fixtures/bam_summary/pois")
cat("pois: rank=", mp$rank, "/", length(coef(mp)),
    " scale=", format(summary(mp)$scale, digits = 10), "\n")

# ---------- rankdef : rank-deficient design ---------------------------------
# An exactly-collinear parametric column (xdup == x): the unpenalized
# parametric block is singular, so mgcv's fitting rank-reveal drops one
# direction (assigns it beta=0) and reports m$rank = p-1. Exercises the
# `Rank: r/p` summary line and oo$rank.est < p on a bam.
set.seed(3)
n <- 60
x <- runif(n)
z <- runif(n)
xdup <- x
y <- rnorm(n, 0.5 + sin(3 * x) + 0.5 * z, sd = 0.5)
dr <- data.frame(y = y, x = x, xdup = xdup, z = z)
mr <- bam(y ~ x + xdup + s(z, k = 6), data = dr)
write.csv(dr, "tests/fixtures/bam_summary/rankdef/data.csv", row.names = FALSE)
.dump(mr, "tests/fixtures/bam_summary/rankdef")
cat("rankdef: rank=", mr$rank, "/", length(coef(mr)), "\n")
