## mgcv's C_gdi2 as gam.fit4 calls it (gam.fit4.r:609) and gam.fit4 itself,
## captured from real fits with extended families, for the gdi2 and gam.fit4
## ports. gam.fit4 is taken from ref/mgcv/R/gam.fit4.r with its .C(C_gdi2,
## ...) routed through a recorder, and installed in mgcv's namespace (with
## gam.fit4 wrapped to record its own arguments and values) for the fits.
## Run from the repository root:
##
##     Rscript tests/scripts/make_gdi2_fixtures.R
##
## tests/fixtures/gdi2/<case>_<k>.txt.gz holds the k-th recorded gdi2 call of
## a fit (first, middle and last); tests/fixtures/gam_fit4/<case>_<k>.txt.gz
## the first and last gam.fit4 call (arguments and the numeric parts of what
## it returns), <case>_gam.txt.gz what the fitted gam reports, and
## data.txt.gz the data. One array per line: its name, dimensions joined by
## commas and the values as C99 hexadecimal floats (sprintf("%a")),
## matrices column-major; "a_" names are the arguments, "o_" the values .C
## returned where gdi2 changed them.
suppressPackageStartupMessages(library(mgcv))
stopifnot(packageVersion("mgcv") == "1.9.4")
outs <- c(gdi2 = "tests/fixtures/gdi2", fit4 = "tests/fixtures/gam_fit4")
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

CAP <- new.env()
CAP$gdi2 <- list()
CAP$fit4 <- list()
.cap_gdi2 <- function(f, ...) {
  a <- list(...)
  o <- .C(f, ...)
  CAP$gdi2[[length(CAP$gdi2) + 1]] <- list(a = a, o = o)
  o
}
src <- parse("ref/mgcv/R/gam.fit4.r", keep.source = FALSE)
is_fit4 <- vapply(src, function(e) is.call(e) && identical(e[[1]], as.name("<-")) &&
                    identical(e[[2]], as.name("gam.fit4")), logical(1))
stopifnot(sum(is_fit4) == 1)
txt <- paste(deparse(src[[which(is_fit4)]][[3]], width.cutoff = 500L), collapse = "\n")
stopifnot(lengths(regmatches(txt, gregexpr(".C(C_gdi2,", txt, fixed = TRUE))) == 1)
txt <- sub(".C(C_gdi2,", ".cap_gdi2(C_gdi2,", txt, fixed = TRUE)
fit4_core <- eval(parse(text = txt)[[1]], envir = asNamespace("mgcv"))
environment(fit4_core) <- asNamespace("mgcv")
fit4_rec <- function(x, y, sp, Eb, UrS = list(), weights = rep(1, NROW(y)), start = NULL,
                     etastart = NULL, mustart = NULL, offset = rep(0, NROW(y)),
                     U1 = diag(ncol(x)), Mp = -1, family = gaussian(), control = gam.control(),
                     deriv = 2, gamma = 1, scale = 1, scoreType = "REML", null.coef = rep(0, ncol(x)),
                     nei = NULL, ...) {
  a <- list(x = x, y = y, sp = sp, Eb = Eb, weights = weights, start = start,
            etastart = etastart, mustart = mustart, offset = offset, U1 = U1, Mp = Mp,
            deriv = deriv, gamma = gamma, scale = scale, scoreType = scoreType,
            null.coef = null.coef, epsilon = control$epsilon, maxit = control$maxit,
            family = family$family, link = family$link, theta = family$getTheta(),
            nUrS = length(UrS))
  for (i in seq_along(UrS)) a[[paste0("UrS", i)]] <- UrS[[i]]
  o <- fit4_core(x, y, sp, Eb, UrS = UrS, weights = weights, start = start,
                 etastart = etastart, mustart = mustart, offset = offset, U1 = U1, Mp = Mp,
                 family = family, control = control, deriv = deriv, gamma = gamma,
                 scale = scale, scoreType = scoreType, null.coef = null.coef, nei = nei, ...)
  keep <- c("coefficients", "residuals", "fitted.values", "linear.predictors", "deviance",
            "null.deviance", "iter", "weights", "working.weights", "z", "converged",
            "boundary", "D1", "D2", "P", "P1", "P2", "REML", "REML1", "REML2", "rV",
            "db.drho", "dw.drho", "dVkk", "ldetS1", "scale.est", "aic", "rank", "K")
  CAP$fit4[[length(CAP$fit4) + 1]] <- list(a = a, o = o[intersect(keep, names(o))],
                                           theta = family$getTheta())
  o
}
assignInNamespace("gam.fit4", fit4_rec, ns = "mgcv")

save_calls <- function(case, m) {
  calls <- CAP$gdi2
  stopifnot(length(calls) > 0)
  pick <- unique(c(1L, (length(calls) + 1L) %/% 2L, length(calls)))
  for (k in seq_along(pick)) {
    cl <- calls[[pick[k]]]
    a <- cl$a; o <- cl$o
    o <- o[!mapply(identical, o, a[names(o)])]
    lst <- c(setNames(a, paste0("a_", names(a))), setNames(o, paste0("o_", names(o))))
    dumpl(outs[["gdi2"]], sprintf("%s_%d.txt.gz", case, k), lst)
  }
  calls <- CAP$fit4
  stopifnot(length(calls) > 0)
  pick <- unique(c(1L, length(calls)))
  for (k in seq_along(pick)) {
    cl <- calls[[pick[k]]]
    lst <- c(setNames(cl$a, paste0("a_", names(cl$a))),
             setNames(cl$o, paste0("o_", names(cl$o))), list(o_theta = cl$theta))
    lst <- lst[!vapply(lst, function(v) is.list(v) || is.function(v), logical(1))]
    dumpl(outs[["fit4"]], sprintf("%s_%d.txt.gz", case, k), lst)
  }
  dumpl(outs[["fit4"]], sprintf("%s_gam.txt.gz", case),
        list(coefficients = m$coefficients, Vp = m$Vp, Ve = m$Ve, Vc = m$Vc,
             edf = m$edf, edf1 = m$edf1, edf2 = m$edf2, hat = m$hat, sig2 = m$sig2,
             scale = m$scale, aic = m$aic, gcv.ubre = m$gcv.ubre,
             null.deviance = m$null.deviance, deviance = m$deviance, rank = m$rank,
             sp = m$sp, full.sp = m$full.sp, hess = m$outer.info$hess,
             grad = m$outer.info$grad, iter = m$outer.info$iter, R = m$R,
             df.residual = m$df.residual, fitted.values = m$fitted.values,
             linear.predictors = m$linear.predictors, weights = m$weights,
             theta = m$family$getTheta(), logLik = as.numeric(logLik(m)), AIC = AIC(m)))
  types <- vapply(CAP$gdi2, function(cl) cl$a$type, integer(1))
  nneg <- vapply(CAP$gdi2, function(cl) sum(cl$a$w < 0), numeric(1))
  cat(case, ":", length(CAP$gdi2), "gdi2 and", length(CAP$fit4), "gam.fit4 calls; type",
      paste(unique(types), collapse = "/"), "; ldet", CAP$gdi2[[1]]$a$ldet,
      "; negative w per call", range(nneg), "\n")
  CAP$gdi2 <- list()
  CAP$fit4 <- list()
}

## --- the fits -------------------------------------------------------------
set.seed(7)
n <- 300
d <- data.frame(x1 = runif(n), x2 = runif(n))
f <- sin(2 * pi * d$x1) + 0.5 * d$x2
d$ynb <- rnbinom(n, size = 3, mu = exp(f / 2 + 0.5))
d$ytw <- rTweedie(exp(f / 3), p = 1.5, phi = 0.8)
d$yt <- f + 0.3 * rt(n, df = 4)
d$ybr <- binomial()$linkinv(f / 2 + rnorm(n) * 0.4)
d$yoc <- cut(f + rnorm(n) * 0.5, c(-Inf, -0.5, 0.5, Inf), labels = FALSE)
d$yzp <- ifelse(runif(n) < 0.3, 0, rpois(n, exp(f / 2 + 0.3)))

m <- gam(ynb ~ s(x1) + s(x2), data = d, family = nb(), method = "REML"); save_calls("nb_reml", m)
m <- gam(ynb ~ s(x1) + s(x2), data = d, family = nb(), method = "ML"); save_calls("nb_ml", m)
m <- gam(ynb ~ s(x1) + s(x2), data = d, family = nb(theta = 3), method = "REML"); save_calls("nb_fixed_reml", m)
m <- gam(ytw ~ s(x1) + s(x2), data = d, family = tw(), method = "REML"); save_calls("tw_reml", m)
m <- gam(yt ~ s(x1) + s(x2), data = d, family = scat(), method = "REML"); save_calls("scat_reml", m)
m <- gam(ybr ~ s(x1) + s(x2), data = d, family = betar(), method = "REML"); save_calls("betar_reml", m)
m <- gam(yoc ~ s(x1) + s(x2), data = d, family = ocat(R = 3), method = "REML"); save_calls("ocat_reml", m)
m <- gam(yzp ~ s(x1) + s(x2), data = d, family = ziP(), method = "REML"); save_calls("zip_reml", m)
## the extended Fellner-Schall optimizer: theta re-estimated in PIRLS
m <- gam(ynb ~ s(x1) + s(x2), data = d, family = nb(), method = "REML", optimizer = "efs")
save_calls("nb_efs", m)
m <- gam(ytw ~ s(x1) + s(x2), data = d, family = tw(), method = "REML", optimizer = "efs")
save_calls("tw_efs", m)
## nlm and optim: gam.fit3 hands each objective's fit to gam.fit4
m <- gam(ynb ~ s(x1) + s(x2), data = d, family = nb(), method = "REML", optimizer = c("outer", "nlm"))
save_calls("nb_nlm", m)
m <- gam(ynb ~ s(x1) + s(x2), data = d, family = nb(), method = "ML", optimizer = c("outer", "optim"))
save_calls("nb_ml_optim", m)
m <- gam(ytw ~ s(x1) + s(x2), data = d, family = tw(), method = "REML", optimizer = c("outer", "optim"))
save_calls("tw_optim", m)
## zero prior weights make gam.fit4's weights zero there: gdi2's type 1
d$w0 <- rep(1, n); d$w0[c(3, 50, 120)] <- 0
m <- gam(ynb ~ s(x1) + s(x2), data = d, family = nb(), weights = w0, method = "REML")
save_calls("nb_reml_w0", m)
dumpl(outs[["fit4"]], "data.txt.gz",
      list(x1 = d$x1, x2 = d$x2, ynb = d$ynb, ytw = d$ytw, yt = d$yt, ybr = d$ybr,
           yoc = d$yoc, yzp = d$yzp, w0 = d$w0))
