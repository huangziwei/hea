## bam(scale=) oracle (plan item P6). bam used to take no scale= arg.
## scale>0 fixes φ at a known value — REML/ML then select sp with φ held fixed
## (so the fit differs from the φ-estimated default). Pinned on a Gaussian fit
## where φ is fixed at 2.5. (scale-known families poisson/binomial only support
## scale=0 in hea — bam's quasi-likelihood φ estimation isn't ported.)
##
## Dumps (under tests/fixtures/bam_scale/): fitted.csv + meta.csv
## (scale, edf_total, sp, intercept est/se) for y ~ z + s(x, k=10), scale=2.5.

suppressMessages(library(mgcv))

dg <- read.csv("tests/fixtures/bam_summary/gauss/data.csv")
m <- bam(y ~ z + s(x, k = 10), data = dg, scale = 2.5)

dir.create("tests/fixtures/bam_scale", showWarnings = FALSE, recursive = TRUE)
writeLines(format(fitted(m), digits = 15), "tests/fixtures/bam_scale/fitted.csv")
s <- summary(m)
meta <- data.frame(
  scale     = m$sig2,
  edf_total = sum(m$edf),
  sp        = m$sp[1],
  int_est   = s$p.table["(Intercept)", "Estimate"],
  int_se    = s$p.table["(Intercept)", "Std. Error"]
)
write.csv(meta, "tests/fixtures/bam_scale/meta.csv", row.names = FALSE)
cat("scale=2.5: sig2=", m$sig2, " edf=", format(sum(m$edf), digits = 9), "\n")
