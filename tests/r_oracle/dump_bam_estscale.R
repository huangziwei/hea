## bam scale-UNKNOWN non-Gaussian φ-estimation oracle (plan item P19).
##
## bam fits a scale-unknown non-Gaussian family (Gamma, inverse Gaussian,
## fixed-p Tweedie, the extended families) by PIRLS over a reduced (R, f)
## problem. hea USED to run the converge-fully ``_outer_newton`` on each frozen
## (R, f) linearisation, which minimises the WORKING-RSS REML — whose (ρ, φ)
## optimum differs from the RESPONSE-deviance optimum mgcv reaches by taking ONE
## POI step then re-linearising (recompute W, z, dev at the new β̂). The gap was
## large: Gamma sp 0.158 vs mgcv-bam 0.205 (3.7×), Tweedie sp 0.207 vs 0.259.
## hea now routes these families through the same one-step POI cadence mgcv uses
## (bgam.fitd), so sp / scale / edf / fitted pin to mgcv-**bam** (NOT gam —
## bam's reduced-(R,f) cadence genuinely differs from gam's full-data fit:
## mgcv-gam Gamma sp 0.579 vs mgcv-bam 0.205).
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
