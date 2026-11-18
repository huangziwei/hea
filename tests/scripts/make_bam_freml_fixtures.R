## bam(method = "fREML") fits for tests/test_bam.py: the data, then per
## fit the coefficients, smoothing parameters, scale (sig2), gcv.ubre, edf,
## edf1, edf2, Vp, Vc and Ve that bam.fit (Gaussian), bgam.fit (PIRLS) or,
## with discrete = TRUE, bgam.fitd produce. Run from the repository root:
##
##     Rscript tests/scripts/make_bam_freml_fixtures.R
##
## data.bin holds n, then x, z and y; glm_data.bin holds n, then x, z and
## the Poisson, binomial and Gamma responses; discrete_data.bin holds n,
## then x, z, zb, the level codes of fac, and the Gaussian, Poisson and
## Tweedie responses y, yp and yt (g and g4, factors of rep(1:20) and
## rep(1:4), are added in the script and in the tests). <name>.bin holds p and the number of
## smoothing parameters, then the values in the order above (matrices
## column-major), then the fitted values.
suppressPackageStartupMessages(library(mgcv))
stopifnot(packageVersion("mgcv") == "1.9.4")
out <- "tests/fixtures/bam_freml"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
set.seed(3)
n <- 400
d <- data.frame(x = runif(n), z = runif(n))
d$y <- sin(2 * pi * d$x) + exp(2 * d$z) / 3 + rnorm(n) * 0.4
writeBin(c(n, d$x, d$z, d$y), file.path(out, "data.bin"))
save_fit <- function(b, nm)
  writeBin(c(length(coef(b)), length(b$sp), as.double(coef(b)), as.double(b$sp), b$sig2,
             b$gcv.ubre, as.double(b$edf), as.double(b$edf1), as.double(b$edf2),
             as.double(b$Vp), as.double(b$Vc), as.double(b$Ve), as.double(fitted(b))),
           file.path(out, paste0(nm, ".bin")))
fits <- list(s_x_s_z = y ~ s(x) + s(z), t2_xz = y ~ t2(x, z))
for (nm in names(fits)) save_fit(bam(fits[[nm]], data = d, method = "fREML"), nm)
save_fit(bam(y ~ s(x) + s(z), data = d, method = "fREML", discrete = TRUE), "gauss_discrete")
save_fit(bam(y ~ s(x) + s(z), data = d, method = "fREML", discrete = TRUE, rho = 0.4),
         "gauss_discrete_ar1")
save_fit(bam(y ~ te(x, z), data = d, method = "fREML", discrete = TRUE), "te_discrete")
save_fit(bam(y ~ s(x) + s(z), data = d, method = "fREML", use.chol = TRUE), "gauss_use_chol")
set.seed(5)
d <- data.frame(x = runif(n), z = runif(n))
eta <- 0.5 + sin(2 * pi * d$x) + d$z
d$yp <- rpois(n, exp(eta))
d$yb <- rbinom(n, 1, binomial()$linkinv(eta - 0.5))
d$yg <- rgamma(n, shape = 5, rate = 5 / exp(eta))
writeBin(c(n, d$x, d$z, d$yp, d$yb, d$yg), file.path(out, "glm_data.bin"))
glm_fits <- list(poisson = list(yp ~ s(x) + s(z), poisson()),
                 binomial = list(yb ~ s(x) + s(z), binomial()),
                 gamma_log = list(yg ~ s(x) + s(z), Gamma(link = "log")))
for (nm in names(glm_fits))
  save_fit(bam(glm_fits[[nm]][[1]], data = d, family = glm_fits[[nm]][[2]], method = "fREML"), nm)
save_fit(bam(yp ~ s(x) + s(z), data = d, family = poisson(), method = "fREML", discrete = TRUE),
         "poisson_discrete")
save_fit(bam(yp ~ s(x) + s(z), data = d, family = poisson(), method = "fREML", use.chol = TRUE),
         "poisson_use_chol")
set.seed(11)
n <- 600
d <- data.frame(x = runif(n), z = runif(n), zb = runif(n) + 0.5)
d$fac <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
eta <- sin(2 * pi * d$x) + exp(2 * d$z) / 3 + as.numeric(d$fac) * 0.3
d$y <- eta + rnorm(n) * 0.4
d$yp <- rpois(n, exp(eta / 2))
d$yt <- rTweedie(exp(eta / 2), p = 1.5, phi = 1)
writeBin(c(n, d$x, d$z, d$zb, as.numeric(d$fac), d$y, d$yp, d$yt),
         file.path(out, "discrete_data.bin"))
discrete_fits <- list(
  d_by_num = quote(bam(y ~ s(x, by = zb) + s(z), data = d, discrete = TRUE)),
  d_by_fac = quote(bam(y ~ s(x, by = fac) + fac, data = d, discrete = TRUE)),
  d_te_tp = quote(bam(y ~ te(x, z, bs = "tp"), data = d, discrete = TRUE)),
  d_te_crtp = quote(bam(y ~ te(x, z, bs = c("cr", "tp")), data = d, discrete = TRUE)),
  d_ti = quote(bam(y ~ ti(x) + ti(z) + ti(x, z), data = d, discrete = TRUE)),
  d_cr_ps = quote(bam(y ~ s(x, bs = "cr") + s(z, bs = "ps"), data = d, discrete = TRUE)),
  d_s2d = quote(bam(y ~ s(x, z), data = d, discrete = TRUE)),
  d_re = quote(bam(y ~ s(x) + s(fac, bs = "re"), data = d, discrete = TRUE)),
  d_cc_k20 = quote(bam(y ~ s(x, k = 20) + s(z, bs = "cc"), data = d, discrete = TRUE)),
  d_nb = quote(bam(yp ~ s(x) + s(z), family = nb(), data = d, discrete = TRUE)),
  d_tw = quote(bam(yt ~ s(x) + s(z), family = tw(), data = d, discrete = TRUE)),
  d_pois_te = quote(bam(yp ~ te(x, z), family = poisson(), data = d, discrete = TRUE)),
  d_te_byfac = quote(bam(y ~ te(x, z, by = fac) + fac, data = d, discrete = TRUE)),
  d_te_bynum = quote(bam(y ~ te(x, z, by = zb), data = d, discrete = TRUE)),
  d_t2 = quote(bam(y ~ t2(x, z), data = d, discrete = TRUE)),
  d_ti_k = quote(bam(y ~ s(x) + s(z) + ti(x, z, k = c(5, 4)), data = d, discrete = TRUE)))
for (nm in names(discrete_fits)) save_fit(eval(discrete_fits[[nm]]), nm)
d$g <- factor(rep(1:20, length.out = n))
save_fit(bam(y ~ s(x) + s(fac, z, bs = "re"), data = d), "re_slope")
save_fit(bam(y ~ s(x) + s(fac, g, bs = "re"), data = d), "re_2fac")
save_fit(bam(y ~ s(x, fac, bs = "fs", k = 5), data = d, discrete = TRUE), "d_fs")
save_fit(bam(y ~ s(x) + s(fac, z, bs = "re"), data = d, discrete = TRUE), "d_re_slope")
save_fit(bam(y ~ s(x) + s(fac, g, bs = "re"), data = d, discrete = TRUE), "d_re_2fac")
d$g4 <- factor(rep(1:4, length.out = n))
save_fit(bam(y ~ s(x) + s(fac, g4, x, bs = "sz"), data = d), "sz_2fac")
save_fit(bam(y ~ s(x) + s(fac, x, bs = "sz"), data = d, discrete = TRUE), "d_sz")
save_fit(bam(y ~ s(x) + s(fac, g4, x, bs = "sz"), data = d, discrete = TRUE), "d_sz_2fac")
save_fit(bam(y ~ s(x) + s(z) + te(x, z), data = d, discrete = TRUE), "d_side_te")
save_fit(bam(y ~ s(x) + te(x, z), data = d, discrete = TRUE), "d_side_te1")
save_fit(bam(y ~ s(x, by = zb) + te(x, z, by = zb), data = d, discrete = TRUE), "d_side_bynum")
save_fit(bam(y ~ s(x) + s(z), data = d, sp = c(0.3, 2), discrete = TRUE), "d_fixsp")
## no free smoothing parameter: n, p, coef, sig2, gcv.ubre, edf, fitted
save_nofree <- function(b, nm)
  writeBin(c(nrow(d), length(coef(b)), as.double(coef(b)), b$sig2, b$gcv.ubre,
             as.double(b$edf), as.double(fitted(b))), file.path(out, paste0(nm, ".bin")))
save_nofree(bam(y ~ s(x) + s(z), data = d, sp = c(0.3, 2)), "fixsp_gauss")
save_nofree(bam(yp ~ s(x) + s(z), data = d, family = poisson(), sp = c(0.3, 2)), "fixsp_pois")
save_nofree(bam(y ~ x + z, data = d), "param_gauss")
save_nofree(bam(yp ~ x + z, data = d, family = poisson()), "param_pois")
set.seed(3)
d <- data.frame(x = d$x, z = d$z)
mu <- plogis(sin(2 * pi * d$x))
d$yb <- rbeta(n, mu * 5, (1 - mu) * 5)
d$yz <- rpois(n, exp(1 + d$x)) * rbinom(n, 1, 0.7)
d$yo <- as.numeric(cut(d$x + d$z + rnorm(n) * 0.3, c(-Inf, 0.6, 1.0, 1.4, Inf)))
writeBin(c(n, d$x, d$z, d$yb, d$yz, d$yo), file.path(out, "efam_data.bin"))
save_resid <- function(b, nm)
  writeBin(c(as.double(fitted(b)), as.double(b$residuals), b$deviance, b$null.deviance),
           file.path(out, paste0(nm, "_resid.bin")))
for (fit in list(list(yb ~ s(x), betar(), "betar"), list(yz ~ s(x), ziP(), "zip"),
                 list(yo ~ s(x) + s(z), ocat(R = 4), "ocat"))) {
  b <- bam(fit[[1]], family = fit[[2]], data = d)
  save_fit(b, fit[[3]]); save_resid(b, fit[[3]])
}
