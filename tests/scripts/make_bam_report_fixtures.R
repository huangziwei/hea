## What mgcv's bam reports, for the bam report tests in tests/test_bam.py:
## Gaussian-identity models on every non-discrete method, with prior weights,
## AR1 errors (with and without AR.start), no intercept and a known scale, and
## Poisson, Gamma (log link) and binomial models on every non-discrete method.
## Run from the repository root:
##
##     Rscript tests/scripts/make_bam_report_fixtures.R
##
## tests/fixtures/bam_report/<case>.txt.gz holds what the fitted bam reports
## and data.txt.gz the data. One array per line: its name, dimensions joined
## by commas and the values as C99 hexadecimal floats (sprintf("%a")),
## matrices column-major; "NULL" marks a component mgcv leaves NULL.
suppressPackageStartupMessages(library(mgcv))
stopifnot(packageVersion("mgcv") == "1.9.4")
out <- "tests/fixtures/bam_report"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
dumpl <- function(file, lst) {
  con <- gzfile(file.path(out, file), "w")
  for (k in names(lst)) {
    v <- lst[[k]]
    if (is.null(v)) { writeLines(paste(k, "NULL"), con); next }
    if (is.logical(v)) v <- as.integer(v)
    d <- dim(v); if (is.null(d)) d <- length(v)
    writeLines(paste(k, paste(d, collapse = ","),
                     paste(sprintf("%a", as.double(v)), collapse = " ")), con)
  }
  close(con)
}

set.seed(23)
n <- 1500
d <- data.frame(x1 = runif(n), x2 = runif(n))
f <- sin(2 * pi * d$x1) + 0.8 * cos(2 * pi * d$x2)
d$y <- f + 0.4 * rnorm(n)
d$w <- 0.5 + (seq_len(n) %% 3) / 2
d$ar <- seq_len(n) %% 500 == 1
d$yp <- rpois(n, exp(f / 2 + 1))
d$yg <- rgamma(n, shape = 3, rate = 3 / exp(f / 3))
d$yb <- rbinom(n, 1, binomial()$linkinv(f))

all_methods <- c("REML", "ML", "P-REML", "P-ML", "GACV.Cp", "fREML", "GCV.Cp")
cases <- list()
for (m in all_methods) cases[[paste0("plain_", m)]] <- list(method = m)
for (m in c("REML", "ML", "fREML", "GCV.Cp"))
  cases[[paste0("w_", m)]] <- list(method = m, weights = d$w)
for (m in c("REML", "ML", "P-REML", "fREML", "GCV.Cp"))
  cases[[paste0("ar_", m)]] <- list(method = m, rho = 0.4)
for (m in c("REML", "ML", "fREML"))
  cases[[paste0("arstart_", m)]] <- list(method = m, rho = 0.4, AR.start = d$ar)
for (m in c("ML", "fREML", "GCV.Cp"))
  cases[[paste0("noint_", m)]] <- list(method = m, noint = TRUE)
for (m in c("REML", "ML", "fREML"))
  cases[[paste0("scale_", m)]] <- list(method = m, scale = 0.16)
for (m in all_methods) {
  cases[[paste0("pois_", m)]] <- list(method = m, family = poisson(), y = "yp")
  cases[[paste0("gamma_", m)]] <- list(method = m, family = Gamma(link = "log"), y = "yg")
  cases[[paste0("binom_", m)]] <- list(method = m, family = binomial(), y = "yb")
}

for (case in names(cases)) {
  a <- cases[[case]]
  resp <- if (is.null(a$y)) "y" else a$y
  rhs <- if (isTRUE(a$noint)) "s(x1) + s(x2) - 1" else "s(x1) + s(x2)"
  a$noint <- NULL; a$y <- NULL
  m <- do.call(bam, c(list(formula = as.formula(paste(resp, "~", rhs)), data = d), a))
  s <- summary(m)
  dumpl(paste0(case, ".txt.gz"),
        list(sp = m$sp, coefficients = m$coefficients, gcv.ubre = m$gcv.ubre,
             sig2 = m$sig2, scale = m$scale, deviance = m$deviance,
             null.deviance = m$null.deviance, aic = m$aic, edf = m$edf,
             edf1 = m$edf1, edf2 = m$edf2, Vp = m$Vp, Ve = m$Ve, Vc = m$Vc,
             fitted.values = m$fitted.values, logLik = as.numeric(logLik(m)),
             AIC = AIC(m), r.sq = s$r.sq, dev.expl = s$dev.expl, rank = m$rank))
}
dumpl("data.txt.gz", list(x1 = d$x1, x2 = d$x2, y = d$y, w = d$w, ar = d$ar,
                          yp = d$yp, yg = d$yg, yb = d$yb))
cat(length(cases), "bam fits\n")
