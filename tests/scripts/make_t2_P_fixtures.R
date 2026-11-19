## gam.setup's G$P (the map from fitted to prediction coefficients) for t2
## smooths, and the t2 block bam builds under modCon = 2, for
## tests/test_smooths.py. Run from the repository root:
##
##     Rscript tests/scripts/make_t2_P_fixtures.R
suppressPackageStartupMessages(library(mgcv))
out <- "tests/fixtures/t2_P"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
set.seed(2)
n <- 100
d <- data.frame(x = runif(n), z = runif(n))
d$y <- sin(3 * d$x) + d$z^2 + rnorm(n) * 0.2
d$w <- runif(n)
writeBin(as.double(gam(y ~ t2(x, z), data = d, fit = FALSE)$P), file.path(out, "P.bin"))
writeBin(as.double(gam(y ~ t2(x, z, k = c(4, 5)) + w, data = d, fit = FALSE)$P),
         file.path(out, "P_param.bin"))
writeBin(as.double(bam(y ~ t2(x, z), data = d, fit = FALSE)$X), file.path(out, "bam_X.bin"))
