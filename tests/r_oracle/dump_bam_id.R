## bam id= smoothing-parameter sharing oracle (plan item P9). bam used to
## REJECT id= (reject_unsupported_smooth_id) — each linked penalty would
## silently get its own λ. Now bam ports gam's working-θ L-matrix layer:
## ONE working smoothing parameter drives several penalties (ρ_full = L·θ),
## bases pooled across the linked smooths (idLinksBases, already in
## materialize_smooths). These oracles pin the *gauge-invariant* fit
## quantities against mgcv **bam** (NOT gam — bam builds bases on the
## mini.mf subsample, so log|S|/log|H| and hence the criterion differ from
## gam by a basis-dependent constant; sp/edf/fitted/scale are well-
## determined and DO pin).
##
## Cases (under tests/fixtures/bam_id/<case>/):
##   gauss_cr : Gaussian, cr basis, two smooths share id=1  (different
##              covariate ranges → pooled-knot acid test)
##   gauss_tp : Gaussian, default tp basis, share id=1
##   byfac    : s(x2, by=fac, id=1) — all by-level blocks share ONE λ
##              (the canonical id idiom) + an unlinked s(x0). NOTE: bam's
##              by-factor basis on the mini.mf subsample diverges from
##              gam/mgcv-bam independently of id (a pre-existing by-factor
##              subsample-parity gap), so the test pins only the id LINKAGE
##              structure here, not the edf/sp values.
##   pois     : Poisson-log, share id=1 (non-discrete PIRLS path)
##   pois_disc: Poisson-log, discrete=TRUE — the discrete POI optimiser, the
##              path whose Newton step gets the L-contraction T'g / T'HT
##
## Each dumps: data.csv, fitted.csv, sp.csv (working m$sp), full_sp.csv
## (m$full.sp), stable.csv (summary s.table, per-smooth edf), meta.csv
## (scale, sp.criterion, n_work, n_slots, n).

suppressMessages(library(mgcv))

root <- "tests/fixtures/bam_id"
dir.create(root, showWarnings = FALSE, recursive = TRUE)

dump <- function(case, m, df) {
  d <- file.path(root, case)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  write.csv(df, file.path(d, "data.csv"), row.names = FALSE)
  writeLines(format(fitted(m), digits = 15), file.path(d, "fitted.csv"))
  writeLines(format(m$sp, digits = 15), file.path(d, "sp.csv"))
  writeLines(format(m$full.sp, digits = 15), file.path(d, "full_sp.csv"))
  s <- summary(m)
  write.csv(s$s.table, file.path(d, "stable.csv"), row.names = TRUE)
  meta <- data.frame(
    scale        = as.numeric(m$sig2),
    sp_criterion = as.numeric(s$sp.criterion),
    n_work       = length(m$sp),
    n_slots      = length(m$full.sp),
    n            = nrow(df)
  )
  write.csv(meta, file.path(d, "meta.csv"), row.names = FALSE)
  cat(sprintf("%-9s sp=%s full.sp=%s scale=%.6g crit=%.6f\n",
              case, paste(format(m$sp, digits = 6), collapse = ","),
              paste(format(m$full.sp, digits = 6), collapse = ","),
              m$sig2, s$sp.criterion))
}

## ---- Gaussian, two covariates on different ranges (set.seed(13)) ---------
set.seed(13)
n <- 250
x0 <- runif(n, 0, 1)
x1 <- runif(n, 0, 3)
y  <- sin(2 * pi * x0) + sin(2 * pi * x1 / 3) + rnorm(n, 0, 0.35)
dg <- data.frame(x0 = x0, x1 = x1, y = y)

dump("gauss_cr",
     bam(y ~ s(x0, bs = "cr", id = 1) + s(x1, bs = "cr", id = 1),
         data = dg, method = "fREML"), dg)
dump("gauss_tp",
     bam(y ~ s(x0, id = 1) + s(x1, id = 1), data = dg, method = "fREML"), dg)

## ---- by=factor single-λ idiom (set.seed(5)) -----------------------------
set.seed(5)
n <- 300
x2  <- runif(n, 0, 1)
x0  <- runif(n, 0, 1)
fac <- sample.int(3, n, replace = TRUE)
fl  <- c(0, 1, 2)[fac]
amp <- c(1, 1.5, 0.5)[fac]
y   <- fl + amp * sin(2 * pi * x2) + cos(2 * pi * x0) + rnorm(n, 0, 0.4)
df  <- data.frame(x2 = x2, x0 = x0,
                  fac = factor(paste0("f", fac), levels = c("f1", "f2", "f3")),
                  y = y)
dump("byfac",
     bam(y ~ fac + s(x2, by = fac, id = 1) + s(x0), data = df,
         method = "fREML"), df)

## ---- Poisson-log, share id=1 (PIRLS path) -------------------------------
set.seed(13)
n <- 250
x0 <- runif(n, 0, 1)
x1 <- runif(n, 0, 3)
eta <- 0.7 * sin(2 * pi * x0) + 0.6 * sin(2 * pi * x1 / 3)
yp  <- rpois(n, exp(eta))
dp  <- data.frame(x0 = x0, x1 = x1, y = yp)
dump("pois",
     bam(y ~ s(x0, id = 1) + s(x1, id = 1), data = dp, family = poisson,
         method = "fREML"), dp)

## NOTE — discrete=TRUE id parity: the discrete POI optimiser's id support
## (the T'g / T'HT Newton-step contraction) is tested in Python against the
## NON-discrete mgcv `pois` reference above, because for these *continuous*
## covariates hea bins by exact unique value (discrete ≡ non-discrete) and
## hea-discrete-id lands on the mgcv NON-discrete id fit (= the mgcv GAM id
## fit). mgcv's OWN discrete=TRUE+id combination instead discretises the
## *pooled* id basis and drifts to a different sp (e.g. 0.0153 → 0.2526 on a
## 12×15 grid) — an mgcv-discrete-internal quirk hea does not (and the gam
## reference does not) reproduce. So there is no mgcv-discrete-id fixture.

cat("bam_id oracle written under", root, "\n")
