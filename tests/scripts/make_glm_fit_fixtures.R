## R's glm on fixed data for tests/test_glm.py's exact pins: per fit the
## coefficients, summary.glm's standard errors and dispersion, the deviance,
## null deviance (refitted by glm.fit when there is an offset), AIC, fitted
## values, vcov, iteration count and predict(se.fit = TRUE)'s standard
## errors on the link scale. Run from the repository root:
##
##     Rscript tests/scripts/make_glm_fit_fixtures.R
##
## Each file holds one array per line: its name, its dimensions joined by
## commas, then its values as C99 hexadecimal floats (sprintf("%a")),
## matrices column-major.
out <- "tests/fixtures/glm_fit"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
dumpl <- function(file, lst) {
  con <- file(file.path(out, file), "w")
  for (k in names(lst)) {
    v <- lst[[k]]
    d <- dim(v); if (is.null(d)) d <- length(v)
    writeLines(paste(k, paste(d, collapse = ","),
                     paste(sprintf("%a", as.double(v)), collapse = " ")), con)
  }
  close(con)
}
set.seed(21)
n <- 120
d <- data.frame(x1 = runif(n), x2 = rnorm(n), g = factor(sample(letters[1:3], n, TRUE)))
d$off <- 0.3 * d$x1
d$yp <- rpois(n, exp(0.2 + d$x1 + 0.3 * d$x2))
d$yb <- rbinom(n, 1, plogis(-0.5 + d$x1 + d$x2))
d$yg <- rgamma(n, shape = 3, rate = 3 / exp(0.5 + 0.4 * d$x1))
d$yn <- 1 + 2 * d$x1 - d$x2 + rnorm(n)
d$w <- runif(n, 0.5, 2)
dumpl("data.txt", list(x1 = d$x1, x2 = d$x2, g = as.numeric(d$g), off = d$off, yp = d$yp,
                       yb = d$yb, yg = d$yg, yn = d$yn, w = d$w))
save <- function(nm, m) {
  s <- summary(m)
  pr <- predict(m, se.fit = TRUE)
  dumpl(paste0(nm, ".txt"), list(
    coef = coef(m), se = s$coefficients[, 2], dispersion = s$dispersion,
    deviance = m$deviance, null_deviance = m$null.deviance, aic = m$aic,
    fitted = fitted(m), vcov = vcov(m), iter = m$iter, pse = pr$se.fit))
}
save("pois", glm(yp ~ x1 + x2 + g, family = poisson, data = d))
save("pois_off", glm(yp ~ x1 + x2 + offset(off), family = poisson, data = d))
save("binom", glm(yb ~ x1 + x2, family = binomial, data = d))
save("gamma", glm(yg ~ x1 + g, family = Gamma(link = "log"), data = d))
save("gauss_w", glm(yn ~ x1 + x2, data = d, weights = w))
save("quasi_mu", glm(yp ~ x1 + x2, family = quasi(link = "log", variance = "mu"), data = d))
save("quasi_power", glm(yg ~ x1 + x2, family = quasi(link = power(1/3), variance = "mu^2"),
                        data = d))
