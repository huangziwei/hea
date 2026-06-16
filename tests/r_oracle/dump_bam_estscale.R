## bam scale-UNKNOWN non-Gaussian φ-estimation oracle (plan item P19).
##
## bam fits a scale-unknown non-Gaussian family (Gamma, inverse Gaussian,
## fixed-p Tweedie, the extended families) by PIRLS over a reduced (R, f)
## problem. mgcv optimises the smoothing parameters + scale on that reduced
## problem with the GAUSSIAN working-model REML (Sl.fit / Sl.fitChol treat the
## IRLS-linearised (R, f) as Gaussian: (nobs-Mp)*log(2*pi*phi) normalisation,
## NO family ls term — the non-Gaussianness lives only in the OUTER loop that
## rebuilds W, z). hea USED to run ``_outer_newton`` on ``_reml`` — the FULL
## non-Gaussian REML carrying the family's saturated-likelihood ls0(phi) term
## (what mgcv-**gam** uses). On the reduced (R, f) that is a DIFFERENT objective
## with a different phi-hat optimum: Tweedie sp 0.207 vs mgcv-bam 0.259, Gamma
## likewise off. hea now routes these families through ``_pi_fit_chol`` /
## ``_fast_reml_fit`` (the Gaussian working REML, mgcv's Sl.fitChol /
## fast.REML.fit), so sp / scale / edf / fitted pin to mgcv-**bam** to ~7
## digits. NB the WRONG-objective story, not a cadence story: mgcv's one-step
## bgam.fitd (discrete=TRUE) and converge-fully bgam.fit (discrete=FALSE) BOTH
## give Tweedie sp 0.258993 — cadence does not move the optimum. (mgcv-gam,
## which DOES use the full non-Gaussian REML, gives a third value: Tweedie sp
## 0.2666, Gamma sp 0.579 — bam's reduced (R,f) genuinely differs from gam.)
##
## Dumps (tests/fixtures/bam_estscale/<case>/): data.csv, fitted.csv,
## meta.csv (sp, scale=sig2, edf_total, n).

suppressMessages(library(mgcv))

root <- "tests/fixtures/bam_estscale"
dir.create(root, showWarnings = FALSE, recursive = TRUE)

dump <- function(case, m, df) {
  d <- file.path(root, case)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  write.csv(df, file.path(d, "data.csv"), row.names = FALSE)
  writeLines(format(fitted(m), digits = 15), file.path(d, "fitted.csv"))
  meta <- data.frame(sp = m$sp[1], scale = m$sig2,
                     edf_total = sum(m$edf), n = nrow(df))
  write.csv(meta, file.path(d, "meta.csv"), row.names = FALSE)
  cat(sprintf("%-8s sp=%.6f scale=%.6f edf=%.5f\n",
              case, m$sp[1], m$sig2, sum(m$edf)))
}

set.seed(21)
n  <- 300
z  <- runif(n)
x  <- runif(n)
mu <- exp(0.6 * sin(2 * pi * x) + 0.4 * z)

## fixed-p Tweedie (regular family, scale estimated)
yt <- rTweedie(mu, p = 1.4, phi = 1.3)
dt <- data.frame(x = x, z = z, y = yt)
dump("tweedie",
     bam(y ~ z + s(x, k = 12), data = dt, family = Tweedie(p = 1.4),
         method = "fREML"), dt)

## Gamma-log (scale estimated) — the largest pre-fix divergence
yg <- rgamma(n, shape = 2, scale = mu / 2)
dg <- data.frame(x = x, z = z, y = yg)
dump("gamma",
     bam(y ~ z + s(x, k = 12), data = dg, family = Gamma(link = "log"),
         method = "fREML"), dg)

cat("bam_estscale oracle written under", root, "\n")
