## Force-fit oracle for the scat bam `force_theta_sp` tests: mgcv's OWN bam()
## re-run at the converged (theta, sp) READ BACK from the committed fixtures
## (so it is consistent to the rounded values the tests load). This force fit
## is NOT the auto fit — the penalised-deviance convergence test (bgam.fitd:678
## reads dev + sum(rSb^2)) stops it in fewer iters at a slightly different beta
## (mgcv force-vs-auto ~3.6e-8 on `factor`). hea's force fit reproduces THIS to
## ~1e-13, so the tests pin to it, not to the auto `fitted.csv`.
suppressMessages(library(mgcv))
force_dump <- function(dir, formula) {
  df <- read.csv(file.path(dir, "data.csv"))
  if (!is.null(df$g)) df$g <- factor(df$g)
  theta <- scan(file.path(dir, "theta.csv"), quiet = TRUE)
  sp <- scan(file.path(dir, "sp.csv"), quiet = TRUE)
  mf <- bam(formula, data = df, family = scat(theta = theta, min.df = 5),
            sp = sp, discrete = TRUE)
  writeLines(format(fitted(mf), digits = 15),
             file.path(dir, "fitted_force.csv"))
  cat(basename(dir), ": force iter=", mf$iter, "\n")
}
force_dump("tests/fixtures/scat_bam/simple", y ~ s(x, k = 10))
force_dump("tests/fixtures/scat_bam/factor", y ~ g + s(x, by = g, k = 10))
