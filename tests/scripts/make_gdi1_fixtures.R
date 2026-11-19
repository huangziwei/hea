## mgcv's C_gdi1 and C_pls_fit1 as gam.fit3 calls them (gam.fit3.r:550 and
## :334, :346): every argument going in and every array coming back,
## captured from real fits, for the gdi1 and pls_fit1 ports. gam.fit3 is
## taken from ref/mgcv/R/gam.fit3.r with those .C calls routed through
## recorders, and installed in mgcv's namespace for the fits.
## Run from the repository root:
##
##     Rscript tests/scripts/make_gdi1_fixtures.R
##
## tests/fixtures/gdi1/<case>_<k>.txt.gz (and pls_fit1/) holds the k-th
## recorded call of a fit; tests/fixtures/gam_fit3/<case>_<k>.txt.gz the first
## and last gam.fit3 call of each fit, its arguments and the numeric parts of
## what it returns (UrS as UrS1, UrS2, ...; the family by name and link),
## <case>_gam.txt.gz what the fitted gam object reports (coefficients, the
## covariances and edfs of gam.fit3.post.proc, scale, aic, score, the outer
## Hessian, ...), and data.txt.gz the data
## (first, middle and last): one array per line, its name, dimensions joined
## by commas and values as C99 hexadecimal floats (sprintf("%a")), matrices
## column-major; "a_" names are the arguments, "o_" the values .C returned
## where gdi1 changed them.
suppressPackageStartupMessages(library(mgcv))
stopifnot(packageVersion("mgcv") == "1.9.4")
outs <- c(gdi1 = "tests/fixtures/gdi1", pls = "tests/fixtures/pls_fit1",
          fit3 = "tests/fixtures/gam_fit3")
for (o in outs) dir.create(o, showWarnings = FALSE, recursive = TRUE)
dumpl <- function(out, file, lst) {
  con <- gzfile(file.path(out, file), "w")
  for (k in names(lst)) {
    v <- lst[[k]]
    if (is.null(v)) { writeLines(paste(k, "NULL"), con); next }
    if (is.logical(v)) v <- as.integer(v)
    if (is.character(v)) { writeLines(paste(k, "CHR", v[1]), con); next }
    d <- dim(v); if (is.null(d)) d <- length(v)
    writeLines(paste(k, paste(d, collapse = ","),
                     paste(sprintf("%a", as.double(v)), collapse = " ")), con)
  }
  close(con)
}

## --- gam.fit3 with its gdi1 call recorded ---------------------------------
CAP <- new.env()
CAP$gdi1 <- list()
CAP$pls <- list()
CAP$fit3 <- list()
.cap_gdi1 <- function(f, ...) {
  a <- list(...)
  o <- .C(f, ...)
  CAP$gdi1[[length(CAP$gdi1) + 1]] <- list(a = a, o = o)
  o
}
.cap_pls <- function(f, ...) {
  a <- list(...)
  o <- .C(f, ...)
  CAP$pls[[length(CAP$pls) + 1]] <- list(a = a, o = o)
  o
}
src <- parse("ref/mgcv/R/gam.fit3.r", keep.source = FALSE)
is_fit3 <- vapply(src, function(e) is.call(e) && identical(e[[1]], as.name("<-")) &&
                    identical(e[[2]], as.name("gam.fit3")), logical(1))
stopifnot(sum(is_fit3) == 1)
txt <- paste(deparse(src[[which(is_fit3)]][[3]], width.cutoff = 500L), collapse = "\n")
stopifnot(grepl(".C(C_gdi1,", txt, fixed = TRUE))
txt <- sub(".C(C_gdi1,", ".cap_gdi1(C_gdi1,", txt, fixed = TRUE)
stopifnot(lengths(regmatches(txt, gregexpr(".C(C_pls_fit1,", txt, fixed = TRUE))) == 2)
txt <- gsub(".C(C_pls_fit1,", ".cap_pls(C_pls_fit1,", txt, fixed = TRUE)
fit3 <- eval(parse(text = txt)[[1]], envir = asNamespace("mgcv"))
environment(fit3) <- asNamespace("mgcv")
fit3_core <- fit3
## gam.fit3 itself recorded: its arguments and what it returns
fit3_rec <- function(x, y, sp, Eb, UrS = list(), weights = rep(1, NROW(y)), start = NULL,
                     etastart = NULL, mustart = NULL, offset = rep(0, NROW(y)),
                     U1 = diag(ncol(x)), Mp = -1, family = gaussian(), control = gam.control(),
                     intercept = TRUE, deriv = 2, gamma = 1, scale = 1, printWarn = TRUE,
                     scoreType = "REML", null.coef = rep(0, ncol(x)), pearson.extra = 0,
                     dev.extra = 0, n.true = -1, Sl = NULL, nei = NULL, ...) {
  a <- list(x = x, y = y, sp = sp, Eb = Eb, weights = weights, start = start,
            etastart = etastart, mustart = mustart, offset = offset, U1 = U1, Mp = Mp,
            intercept = intercept, deriv = deriv, gamma = gamma, scale = scale,
            scoreType = scoreType, null.coef = null.coef, pearson.extra = pearson.extra,
            dev.extra = dev.extra, n.true = n.true, epsilon = control$epsilon,
            maxit = control$maxit, family = family$family, link = family$link,
            nUrS = length(UrS))
  for (i in seq_along(UrS)) a[[paste0("UrS", i)]] <- UrS[[i]]
  o <- fit3_core(x, y, sp, Eb, UrS = UrS, weights = weights, start = start,
                 etastart = etastart, mustart = mustart, offset = offset, U1 = U1, Mp = Mp,
                 family = family, control = control, intercept = intercept, deriv = deriv,
                 gamma = gamma, scale = scale, printWarn = printWarn, scoreType = scoreType,
                 null.coef = null.coef, pearson.extra = pearson.extra, dev.extra = dev.extra,
                 n.true = n.true, Sl = Sl, nei = nei, ...)
  keep <- c("coefficients", "residuals", "fitted.values", "linear.predictors", "deviance",
            "null.deviance", "iter", "weights", "working.weights", "z", "converged",
            "boundary", "D1", "D2", "P", "P1", "P2", "trA", "trA1", "trA2", "GCV", "GCV1",
            "GCV2", "UBRE", "UBRE1", "UBRE2", "REML", "REML1", "REML2", "rV", "db.drho",
            "dw.drho", "dVkk", "ldetS1", "scale.est", "reml.scale", "aic", "rank", "K")
  CAP$fit3[[length(CAP$fit3) + 1]] <- list(a = a, o = o[keep])
  o
}
assignInNamespace("gam.fit3", fit3_rec, ns = "mgcv")

save_calls <- function(case) {
  for (kind in c("gdi1", "pls")) {
    calls <- CAP[[kind]]
    stopifnot(length(calls) > 0)
    pick <- unique(c(1L, (length(calls) + 1L) %/% 2L, length(calls)))
    for (k in seq_along(pick)) {
      cl <- calls[[pick[k]]]
      a <- cl$a; o <- cl$o
      o <- o[!mapply(identical, o, a[names(o)])]
      lst <- c(setNames(a, paste0("a_", names(a))), setNames(o, paste0("o_", names(o))))
      dumpl(outs[[kind]], sprintf("%s_%d.txt.gz", case, k), lst)
    }
  }
  calls <- CAP$fit3
  stopifnot(length(calls) > 0)
  pick <- unique(c(1L, length(calls)))
  for (k in seq_along(pick)) {
    cl <- calls[[pick[k]]]
    lst <- c(setNames(cl$a, paste0("a_", names(cl$a))), setNames(cl$o, paste0("o_", names(cl$o))))
    lst <- lst[!vapply(lst, function(v) is.list(v) || is.function(v), logical(1))]
    dumpl(outs[["fit3"]], sprintf("%s_%d.txt.gz", case, k), lst)
  }
  m <- LAST
  dumpl(outs[["fit3"]], sprintf("%s_gam.txt.gz", case),
        list(coefficients = m$coefficients, Vp = m$Vp, Ve = m$Ve, Vc = m$Vc,
             edf = m$edf, edf1 = m$edf1, edf2 = m$edf2, hat = m$hat, sig2 = m$sig2,
             scale = m$scale, aic = m$aic, gcv.ubre = m$gcv.ubre,
             null.deviance = m$null.deviance, deviance = m$deviance, rank = m$rank,
             sp = m$sp, full.sp = m$full.sp, hess = m$outer.info$hess,
             grad = m$outer.info$grad, iter = m$outer.info$iter, R = m$R,
             df.residual = m$df.residual, fitted.values = m$fitted.values,
             linear.predictors = m$linear.predictors, residuals = m$residuals,
             weights = m$weights, logLik = as.numeric(logLik(m)), AIC = AIC(m)))
  calls <- CAP$gdi1
  nneg <- vapply(calls, function(cl) sum(cl$a$w < 0), numeric(1))
  cat(case, ":", length(calls), "gdi1 and", length(CAP$pls), "pls_fit1 calls; fisher",
      calls[[1]]$a$fisher, "fixed.penalty", calls[[1]]$a$fixed.penalty,
      "negative w per call", range(nneg), "\n")
  CAP$gdi1 <- list()
  CAP$pls <- list()
  CAP$fit3 <- list()
}

## --- the fits -------------------------------------------------------------
set.seed(3)
n <- 300
d <- data.frame(x1 = runif(n), x2 = runif(n), x3 = runif(n))
f <- sin(2 * pi * d$x1) + 0.5 * d$x2
d$y <- f + rnorm(n) * 0.3
d$yp <- rpois(n, exp(f / 2))
d$yb <- rbinom(n, 1, binomial()$linkinv(f))

LAST <- (gam(y ~ s(x1) + s(x2), data = d, method = "REML"));                 save_calls("gauss_reml")
LAST <- (gam(yp ~ s(x1) + s(x2), data = d, family = poisson, method = "REML")); save_calls("pois_reml")
LAST <- (gam(yb ~ s(x1) + s(x2), data = d, family = binomial, method = "ML")); save_calls("binom_ml")
LAST <- (gam(y ~ s(x1, bs = "cr") + s(x2, bs = "cr"), data = d, method = "ML")); save_calls("gauss_ml_cr")
## s(x2) has linear truth: its sp runs to a large-lambda flat direction
LAST <- (gam(yp ~ s(x1) + s(x2) + s(x3), data = d, family = poisson, method = "REML")); save_calls("pois_flat")
## a paraPen penalty at a fixed sp, reaching gdi1 through lsp0
Pfix <- list(X3 = list(diag(2), sp = 0.5))
d$X3 <- cbind(d$x3, d$x3^2)
LAST <- (gam(y ~ s(x1) + X3, data = d, paraPen = Pfix, method = "REML")); save_calls("gauss_parapen_sp")
## the same through one column, as hea's paraPen takes it
LAST <- (gam(y ~ s(x1) + x3, data = d, paraPen = list(x3 = list(matrix(1), sp = 0.5)),
             method = "REML")); save_calls("gauss_parapen1_sp")
## min.sp puts a fixed penalty H beside the estimated ones (fixed.penalty = 1)
LAST <- (gam(y ~ s(x1) + s(x2), data = d, min.sp = c(0.01, 0.1), method = "REML")); save_calls("gauss_min_sp")
## non-canonical links: full Newton (fisher = 0), with negative weights
d$yg <- rgamma(n, shape = 3, rate = 3 / exp(f / 2))
LAST <- (gam(yp ~ s(x1) + s(x2), data = d, family = poisson(link = "sqrt"), method = "REML")); save_calls("pois_sqrt_reml")
LAST <- (gam(yb ~ s(x1) + s(x2), data = d, family = binomial(link = "cloglog"), method = "ML")); save_calls("binom_cloglog_ml")
LAST <- (gam(yg ~ s(x1) + s(x2), data = d, family = Gamma(link = "identity"), method = "REML")); save_calls("gamma_id_reml")
LAST <- (gam(yg ~ s(x1) + s(x2), data = d, family = Gamma(link = "identity"), method = "ML")); save_calls("gamma_id_ml")
## GCV/UBRE (REML = 0): tr(A) and its derivatives through get_trA2
LAST <- (gam(yp ~ s(x1) + s(x2), data = d, family = poisson, method = "GCV.Cp")); save_calls("pois_ubre")
LAST <- (gam(yg ~ s(x1) + s(x2), data = d, family = Gamma(link = "log"), method = "GCV.Cp")); save_calls("gamma_log_gcv")
## small problems, ranks 8 to 11 and 30 observations: clang vectorizes some
## of gdi1's reductions only above a length threshold
LAST <- (gam(y ~ s(x1, k = 5) + s(x2, k = 5), data = d, method = "REML")); save_calls("gauss_reml_q9")
LAST <- (gam(yp ~ s(x1, k = 6) + s(x2, k = 5), data = d, family = poisson, method = "REML")); save_calls("pois_reml_q10")
LAST <- (gam(yb ~ s(x1, k = 6) + s(x2, k = 6), data = d, family = binomial, method = "ML")); save_calls("binom_ml_q11")
LAST <- (gam(yg ~ s(x1, k = 4) + s(x2, k = 5), data = d, family = Gamma(link = "identity"), method = "ML")); save_calls("gamma_id_ml_q8")
LAST <- (gam(yp ~ s(x1, k = 5) + s(x2, k = 5), data = d, family = poisson, method = "GCV.Cp")); save_calls("pois_ubre_q9")
LAST <- (gam(yg ~ s(x1, k = 6) + s(x2, k = 5), data = d, family = Gamma(link = "log"), method = "GCV.Cp")); save_calls("gamma_log_gcv_q10")
## a parametric column exactly twice another: gdi1's rank detection zeroes
## one (the doubled column's larger norm leads the pivoting, so no tie)
d$xc <- 2 * d$x1
LAST <- (gam(y ~ x1 + xc + s(x2), data = d, method = "REML")); save_calls("gauss_rankdef")
## smooths sharing one sp (id): newton works on L's single column
LAST <- (gam(y ~ s(x1, id = 1) + s(x2, id = 1), data = d, method = "REML")); save_calls("gauss_reml_id")
## the other outer optimizers: nlm, optim and the extended Fellner-Schall
LAST <- (gam(y ~ s(x1) + s(x2), data = d, method = "REML", optimizer = c("outer", "nlm"))); save_calls("gauss_reml_nlm")
LAST <- (gam(yp ~ s(x1) + s(x2), data = d, family = poisson, method = "REML",
             optimizer = c("outer", "optim"))); save_calls("pois_reml_optim")
## three parameters (two sp and the log scale): L-BFGS-B's 2*col-term dots
## reach five terms
LAST <- (gam(y ~ s(x1) + s(x2), data = d, method = "REML",
             optimizer = c("outer", "optim"))); save_calls("gauss_reml_optim")
LAST <- (gam(yg ~ s(x1) + s(x2), data = d, family = Gamma(link = "log"), method = "GCV.Cp",
             optimizer = c("outer", "nlm"))); save_calls("gamma_log_gcv_nlm")
LAST <- (gam(y ~ s(x1) + s(x2), data = d, method = "REML", optimizer = "efs")); save_calls("gauss_reml_efs")
LAST <- (gam(yp ~ s(x1) + s(x2), data = d, family = poisson, method = "REML", optimizer = "efs")); save_calls("pois_reml_efs")
## bam's compressed gam(G=G) refit on (qrx$R, qrx$f) (bam.r:1722-1733)
LAST <- (bam(y ~ s(x1) + s(x2), data = d, method = "ML")); save_calls("bam_gauss_ml")
## every sp fixed: newton over the log scale alone, or one deriv-0 fit
LAST <- (gam(y ~ s(x1) + s(x2), data = d, method = "REML", sp = c(0.5, 20))); save_calls("gauss_reml_fixsp")
LAST <- (gam(yp ~ s(x1) + s(x2), data = d, family = poisson, method = "REML", sp = c(0.5, 20))); save_calls("pois_reml_fixsp")
LAST <- (gam(yg ~ s(x1) + s(x2), data = d, family = Gamma(link = "log"), method = "GCV.Cp",
             sp = c(0.5, 20))); save_calls("gamma_log_gcv_fixsp")
## no smooths: newton over the log scale alone, or one fit
LAST <- (gam(y ~ x1 + x2, data = d, method = "REML")); save_calls("gauss_reml_nosmooth")
LAST <- (gam(yp ~ x1 + x2, data = d, family = poisson, method = "REML")); save_calls("pois_reml_nosmooth")
LAST <- (gam(yg ~ x1 + x2, data = d, family = Gamma(link = "log"), method = "GCV.Cp"))
save_calls("gamma_log_gcv_nosmooth")
d30 <- d[1:30, ]
LAST <- (gam(y ~ s(x1, k = 5) + s(x2, k = 5), data = d30, method = "REML")); save_calls("gauss_reml_n30")
LAST <- (gam(yp ~ s(x1, k = 5) + s(x2, k = 5), data = d30, family = poisson, method = "GCV.Cp")); save_calls("pois_ubre_n30")
LAST <- (gam(yg ~ s(x1, k = 5) + s(x2, k = 5), data = d30, family = Gamma(link = "identity"), method = "REML")); save_calls("gamma_id_reml_n30")
## the data the fits above used (the first 30 rows for the n30 cases)
dumpl(outs[["fit3"]], "data.txt.gz",
      list(x1 = d$x1, x2 = d$x2, x3 = d$x3, y = d$y, yp = d$yp, yb = d$yb, yg = d$yg,
           xc = d$xc))
