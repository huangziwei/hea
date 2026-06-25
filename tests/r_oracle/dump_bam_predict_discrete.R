## predict.bamd oracle for hea.models.bam (F1 — discrete predict path).
##
## hea's bam.predict used to route discrete fits through predict.gam (exact
## basis evaluation). mgcv routes them through predict.bamd (bam.r:1421/1773),
## which RE-DISCRETISES (bins) the covariates of newdata to a grid and gathers
## the prediction via the discrete kernels (Xbd / diagXVXd). For a continuous
## covariate the binning is lossy, so exact-eval != predict.bamd. These oracles
## pin hea's ported predict.bamd against mgcv.
##
## To get a LOSSY binning that hea reproduces (hea matches mgcv-bam tightly only
## at the DEFAULT discretisation — a custom discrete=m is a separate fit-grid
## concern), we use the default discrete=TRUE (1-D: 1000 grid levels) with
## n=3000 continuous-x rows: the grid rounds 3000 -> 1000, so predict.bamd's
## binned gather differs from exact basis eval, and `lpmatrix %*% coef ==
## fitted` (which exact-eval breaks). The novel grid is 2000 points (> 1000),
## so it also rounds, exercising the re-discretise path.
##
## Dumps under tests/fixtures/bam_predict_discrete/{gauss,pois}/:
##   data.csv                x, y (reproduce the fit in python)
##   coef.csv, Vp.csv        mgcv coef + Bayesian cov (gate the fit match)
##   train_link.csv          predict(m, type="link")              [n]
##   train_fitted.csv        predict(m, type="response")          [n]
##   train_se.csv            predict(m, type="link", se.fit=T)$se  [n]
##   train_lp.csv            predict(m, type="lpmatrix")          [n x p]
##   nd_x.csv                novel grid x                          [ng]
##   nd_link.csv             predict(m, nd, type="link")           [ng]
##   nd_resp.csv             predict(m, nd, type="response")       [ng]
##   nd_se.csv               predict(m, nd, type="link", se.fit=T)$se [ng]
##   nd_resp_se.csv          predict(m, nd, type="response", se.fit=T)$se [ng]
##   nd_lp.csv               predict(m, nd, type="lpmatrix")       [ng x p]

suppressMessages(library(mgcv))

base <- "/Users/ziweih/Works/hea/tests/fixtures/bam_predict_discrete"

dump_one <- function(tag, fam) {
  set.seed(if (tag == "gauss") 11 else 12)
  n <- 3000
  x <- runif(n)
  if (tag == "gauss") {
    y <- sin(2 * pi * x) + rnorm(n, sd = 0.3)
  } else {
    y <- rpois(n, exp(0.8 * sin(2 * pi * x) + 0.2))
  }
  dat <- data.frame(x = x, y = y)
  m <- bam(y ~ s(x, k = 15), family = fam, data = dat, discrete = TRUE)

  d <- file.path(base, tag)
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  W <- function(obj, nm) write.table(obj, file.path(d, nm), quote = FALSE,
                                     row.names = FALSE, col.names = FALSE)

  write.csv(dat, file.path(d, "data.csv"), row.names = FALSE)
  W(format(as.numeric(coef(m)), digits = 15), "coef.csv")
  W(m$Vp, "Vp.csv")

  ## training-data predictions (newdata omitted -> predict.bamd on object$model)
  W(format(as.numeric(predict(m, type = "link")), digits = 15), "train_link.csv")
  W(format(as.numeric(predict(m, type = "response")), digits = 15), "train_fitted.csv")
  W(format(as.numeric(predict(m, type = "link", se.fit = TRUE)$se.fit), digits = 15),
    "train_se.csv")
  W(predict(m, type = "lpmatrix"), "train_lp.csv")

  ## novel newdata: dense grid spanning the training range (> 1000 -> rounds)
  ng <- 2000
  xg <- seq(min(x), max(x), length.out = ng)
  nd <- data.frame(x = xg)
  W(format(xg, digits = 15), "nd_x.csv")
  W(format(as.numeric(predict(m, nd, type = "link")), digits = 15), "nd_link.csv")
  W(format(as.numeric(predict(m, nd, type = "response")), digits = 15), "nd_resp.csv")
  W(format(as.numeric(predict(m, nd, type = "link", se.fit = TRUE)$se.fit), digits = 15),
    "nd_se.csv")
  W(format(as.numeric(predict(m, nd, type = "response", se.fit = TRUE)$se.fit), digits = 15),
    "nd_resp_se.csv")
  W(predict(m, nd, type = "lpmatrix"), "nd_lp.csv")

  cat(sprintf("%-6s p=%d  edf=%.4f  sp=%.5g\n", tag, length(coef(m)),
              sum(m$edf), m$sp))
}

dump_one("gauss", gaussian())
dump_one("pois", poisson())
cat("predict.bamd discrete oracle dumped.\n")
