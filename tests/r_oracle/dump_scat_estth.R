## Generates the oracle for ``hea.bam._estimate_theta`` against mgcv's
## ``estimate.theta`` (R/efam.r:5-96) on a Scat family.
##
## Inputs are heavy-tailed (μ is a smooth, residuals are draws from
## t(5)·0.5) so the converged ν is finite (~13) — staying out of the
## ν → ∞ flat-tail region where Newton iterates don't agree to the last
## digit. Two starting points (``near`` close to optimum, ``far`` from
## opposite side) cover the basin of attraction.
##
## Pinned outputs (under tests/fixtures/scat_estth/):
##   inputs.csv     — (y, mu) probe of length n
##   near_init.csv  — log-internal starting θ near the optimum
##   near_out.csv   — converged θ from estimate.theta
##   far_init.csv   — log-internal starting θ on the other side
##   far_out.csv    — converged θ from estimate.theta
##
## Re-run after upgrading mgcv to refresh.

suppressMessages(library(mgcv))
set.seed(99)
n <- 200
y <- rt(n, df = 5) * 0.5 + sin(seq_len(n) / 4)
mu <- sin(seq_len(n) / 4) + rnorm(n, sd = 0.05)

dir <- "tests/fixtures/scat_estth"
dir.create(dir, showWarnings = FALSE, recursive = TRUE)
write.csv(data.frame(y = y, mu = mu),
          file.path(dir, "inputs.csv"), row.names = FALSE)

near_init <- c(0.4, -0.3)
fam <- scat(min.df = 4)
fam$putTheta(near_init)
near_out <- mgcv:::estimate.theta(fam$getTheta(), fam, y, mu,
                                  scale = 1, wt = rep(1, n), tol = 1e-7)
writeLines(format(near_init, digits = 15), file.path(dir, "near_init.csv"))
writeLines(format(near_out, digits = 15), file.path(dir, "near_out.csv"))

far_init <- c(-1.0, 0.5)
fam2 <- scat(min.df = 4)
fam2$putTheta(far_init)
far_out <- mgcv:::estimate.theta(fam2$getTheta(), fam2, y, mu,
                                 scale = 1, wt = rep(1, n), tol = 1e-7)
writeLines(format(far_init, digits = 15), file.path(dir, "far_init.csv"))
writeLines(format(far_out, digits = 15), file.path(dir, "far_out.csv"))
