## mgcv's magic for tests/test_magic.py: the C engine called through .C on
## fixed inputs (c_<case>.txt), the R wrapper magic() with magic.post.proc
## (r_<case>.txt), gam's Gaussian additive GCV/UBRE fits by am.fit
## (gam_<case>.txt) and bam's GCV.Cp fits (bam_<case>.txt) on the data in
## data.txt, plus gam null deviances refitted by glm with an offset and
## bam id-linked fREML coefficients. Run from the repository root:
##
##     Rscript tests/scripts/make_magic_fixtures.R
##
## Each file holds one array per line: its name, its dimensions joined by
## commas, then its values as C99 hexadecimal floats (sprintf("%a")),
## matrices column-major; "NULL" in place of the dimensions marks an absent
## argument.
suppressPackageStartupMessages(library(mgcv))
stopifnot(packageVersion("mgcv") == "1.9.4")
out <- "tests/fixtures/magic"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
dumpl <- function(file, lst) {
  con <- file(file.path(out, file), "w")
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

## --- penalized regression problems for the engine and the wrapper --------
problem <- function(seed, n, ks) {
  set.seed(seed)
  d <- data.frame(x1 = runif(n), x2 = runif(n), x3 = runif(n))
  y <- sin(2 * pi * d$x1) + d$x2^2 + 0.5 * d$x3 + rnorm(n) * 0.3
  bs <- c("tp", "cr", "ps")
  sm <- lapply(seq_along(ks), function(i)
    smoothCon(s(x, k = ks[i], bs = bs[i]), data = data.frame(x = d[[i]]),
              absorb.cons = TRUE)[[1]])
  X <- cbind(1, do.call(cbind, lapply(sm, function(s) s$X)))
  off <- cumsum(c(2, sapply(sm, function(s) ncol(s$X))))[seq_along(sm)]
  list(y = y, X = X, S = lapply(sm, function(s) s$S[[1]]), off = off,
       rank = sapply(sm, function(s) s$rank))
}

## C engine: mgcv's magic.c with penalty roots from mroot(S, rank, "chol")
engine <- function(name, P, sp, def.sp, H = NULL, L = NULL, lsp0 = NULL, gamma = 1,
                   scale = 1, gcv = TRUE, step.half = 25,
                   rank.tol = .Machine$double.eps^0.5, tol = 1e-6, extra.rss = 0,
                   n.score = length(P$y)) {
  q <- ncol(P$X); n <- length(P$y); n.p <- length(P$S)
  Si <- c(); cS <- c()
  for (i in seq_len(n.p)) {
    B <- mroot(P$S[[i]], rank = P$rank[i], method = "chol")
    R <- matrix(0, q, ncol(B)); R[P$off[i]:(P$off[i] + nrow(B) - 1), ] <- B
    Si <- c(Si, R); cS <- c(cS, ncol(B))
  }
  icontrol <- c(as.integer(gcv), n, q, as.integer(!is.null(H)), n.p, step.half,
                if (is.null(L)) -1L else ncol(L))
  Lc <- if (is.null(L)) diag(n.p) else L
  if (is.null(lsp0)) lsp0 <- rep(0, nrow(Lc))
  um <- .C(mgcv:::C_magic, as.double(P$y), X = as.double(P$X), sp = as.double(sp),
           as.double(def.sp), as.double(Si), as.double(H), as.double(Lc),
           lsp0 = as.double(lsp0), score = as.double(gamma), scale = as.double(scale),
           info = as.integer(icontrol), as.integer(cS), as.double(rank.tol),
           rms.grad = as.double(tol), b = as.double(rep(0, q)), rV = double(q * q),
           as.double(extra.rss), as.integer(n.score), as.integer(1))
  rank <- um$info[1]
  dumpl(paste0("c_", name, ".txt"), list(
    y = P$y, X = P$X, sp0 = sp, def_sp = def.sp, S = matrix(Si, q), cS = cS, H = H,
    L = L, lsp0 = lsp0, gamma = gamma, scale_in = scale, gcv = gcv,
    max_half = step.half, rank_tol = rank.tol, tol = tol, norm_const = extra.rss,
    n_score = n.score, o_b = um$b, o_sp = um$sp, o_score = um$score,
    o_scale = um$scale, o_info = um$info[1:5], o_rms = um$rms.grad,
    o_rV = matrix(um$rV[1:(rank * q)], q, rank), o_R = matrix(um$X[1:q^2], q, q)))
}
P <- problem(1, 200, c(10, 8))
engine("gcv_auto", P, sp = rep(-1, 2), def.sp = rep(-1, 2))
engine("ubre_auto", P, sp = rep(-1, 2), def.sp = rep(-1, 2), scale = 0.09, gcv = FALSE)
engine("gcv_given", P, sp = c(0.5, 3), def.sp = c(1, 1))
engine("gcv_flat", P, sp = c(1e-9, 1e9), def.sp = c(0.2, 0.7))
engine("gcv_gamma", P, sp = rep(-1, 2), def.sp = rep(-1, 2), gamma = 1.4,
       extra.rss = 2.5, n.score = 230)
P3 <- problem(2, 150, c(9, 7, 6))
L <- matrix(c(1, 0, 0, 0, 1, 1), 3, 2)
engine("gcv_L", P3, sp = c(0.3, 2), def.sp = c(0.3, 2), L = L, lsp0 = c(0.1, -0.2, 0.3))
engine("ubre_L", P3, sp = c(0.3, 2), def.sp = c(0.3, 2), L = L, lsp0 = c(0, 0, 0),
       scale = 0.1, gcv = FALSE)
H3 <- diag(c(0, rep(0.01, ncol(P3$X) - 1)))
engine("gcv_H", P3, sp = rep(-1, 3), def.sp = rep(-1, 3), H = H3)
P4 <- problem(4, 60, c(5, 4))
engine("small", P4, sp = rep(-1, 2), def.sp = rep(-1, 2))
engine("L0", P4, sp = numeric(0), def.sp = numeric(0), L = matrix(0, 2, 0),
       lsp0 = c(0.5, 1.5))
engine("big", problem(5, 400, c(25, 20, 12)), sp = rep(-1, 3), def.sp = rep(-1, 3))

## R wrapper: magic(y, X, sp, S, off, ...) and magic.post.proc
wrapper <- function(name, P, sp, ..., w = NULL, pp = TRUE) {
  args <- list(...)
  m <- magic(P$y, P$X, sp, P$S, P$off, w = w, ...)
  res <- list(y = P$y, X = P$X, sp_in = sp, off = P$off - 1, nS = length(P$S), w = w)
  for (i in seq_along(P$S)) res[[paste0("S", i)]] <- P$S[[i]]
  for (k in names(args))
    res[[paste0("a_", gsub(".", "_", k, fixed = TRUE))]] <-
      if (is.list(args[[k]])) unlist(args[[k]]) else args[[k]]
  gi <- m$gcv.info
  res <- c(res, list(o_b = m$b, o_scale = m$scale, o_score = m$score, o_sp = m$sp,
                     o_sp_full = m$sp.full, o_R = m$R, o_rV = m$rV,
                     o_info = c(gi$full.rank, gi$rank, gi$fully.converged,
                                gi$hess.pos.def, gi$iter, gi$score.calls),
                     o_rms = gi$rms.grad))
  if (pp) {
    mv <- magic.post.proc(P$X, m, w = if (is.null(w)) NULL else w^2)
    res <- c(res, list(p_Ve = mv$Ve, p_Vb = mv$Vb, p_hat = mv$hat, p_edf = mv$edf,
                       p_edf1 = mv$edf1, p_F = mv$F))
  }
  dumpl(paste0("r_", name, ".txt"), res)
}
Q <- problem(1, 200, c(10, 8))
wrapper("chol", Q, c(-1, -1), rank = Q$rank)
wrapper("svd", Q, c(-1, -1))
set.seed(11); wv <- sqrt(runif(200, 0.5, 2))
wrapper("w", Q, c(-1, -1), rank = Q$rank, w = wv)
wrapper("ubre", Q, c(-1, -1), rank = Q$rank, scale = 0.1, gcv = FALSE)
wrapper("given", Q, c(0.3, 4), rank = Q$rank)
wrapper("control", Q, c(-1, -1), rank = Q$rank, gamma = 1.4, extra.rss = 3, n.score = 250,
        control = list(tol = 1e-7, step.half = 15, rank.tol = .Machine$double.eps^0.5))
Q3 <- problem(2, 150, c(9, 7, 6))
wrapper("L", Q3, c(-1, -1), rank = Q3$rank, L = L, lsp0 = c(0.1, -0.2, 0.3))
q3 <- ncol(Q3$X)
wrapper("C", Q3, c(-1, -1, -1), rank = Q3$rank,
        C = rbind(colMeans(Q3$X), c(1, rep(0, q3 - 1))), pp = FALSE)
wrapper("H", Q3, c(-1, -1, -1), rank = Q3$rank, H = diag(c(0, rep(0.01, q3 - 1))),
        ridge.parameter = 0.05)
wrapper("pad", problem(4, 12, c(10, 8)), c(-1, -1), rank = c(8, 6), pp = FALSE)
wrapper("C0", Q, c(-1, -1), rank = Q$rank, C = matrix(0, 0, ncol(Q$X)), w = wv)

## --- gam's am.fit and bam's GCV.Cp on one data set ----------------------
set.seed(7)
n <- 250
d <- data.frame(x1 = runif(n), x2 = runif(n), x3 = runif(n))
d$fac <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
d$y <- sin(2 * pi * d$x1) + d$x2^2 + 0.3 * as.numeric(d$fac) + rnorm(n) * 0.3
d$w <- runif(n, 0.5, 2); d$w[c(3, 17, 40)] <- 0
d$off <- 0.2 * d$x3
d$yp <- rpois(n, exp(0.5 + sin(2 * pi * d$x1) + d$x2))
dumpl("data.txt", list(x1 = d$x1, x2 = d$x2, x3 = d$x3, fac = as.numeric(d$fac), y = d$y,
                       w = d$w, off = d$off, yp = d$yp))
fitted_model <- function(prefix, nm, b, hat = TRUE) {
  mc <- b$mgcv.conv
  dumpl(paste0(prefix, "_", nm, ".txt"), list(
    coef = coef(b), sp = b$sp, full_sp = b$full.sp, sig2 = b$sig2, edf = b$edf,
    edf1 = b$edf1, Vp = b$Vp, Ve = b$Ve, hat = if (hat) b$hat else NULL,
    gcv = b$gcv.ubre, aic = b$aic, deviance = b$deviance,
    null_deviance = b$null.deviance, fitted = fitted(b), rank = b$rank,
    conv = if (is.null(mc$rms.grad)) NULL else
      c(mc$iter, mc$fully.converged, mc$hess.pos.def, mc$rms.grad)))
}
g <- function(nm, b) fitted_model("gam", nm, b)
g("s2", gam(y ~ s(x1) + s(x2), data = d))
g("w", gam(y ~ s(x1) + s(x2), data = d, weights = w))
g("off", gam(y ~ s(x1) + s(x2) + offset(off), data = d))
g("ubre", gam(y ~ s(x1) + s(x2), data = d, scale = 0.1))
g("id", gam(y ~ s(x1, id = 1) + s(x2, id = 1), data = d))
g("pfix", gam(y ~ s(x1) + s(x2) + s(x3), data = d, sp = c(-1, 0.5, -1)))
g("afix", gam(y ~ s(x1) + s(x2), data = d, sp = c(0.1, 2)))
g("zsp", gam(y ~ s(x1) + s(x2), data = d, sp = c(0, 2)))
g("param", gam(y ~ x1 + x2 + fac, data = d))
g("te", gam(y ~ te(x1, x2), data = d))
g("gamma", gam(y ~ s(x1) + s(x2), data = d, gamma = 1.4))
g("ridge", gam(y ~ s(x1) + s(x2), data = d, control = gam.control(irls.reg = 0.1)))
g("by", gam(y ~ fac + s(x1, by = fac), data = d))
g("cr3", gam(y ~ s(x1, bs = "cr") + s(x2, bs = "cr") + s(x3, bs = "cr"), data = d))
bm <- function(nm, b) fitted_model("bam", nm, b, hat = FALSE)
bm("s2", bam(y ~ s(x1) + s(x2), data = d, method = "GCV.Cp"))
bm("w", bam(y ~ s(x1) + s(x2), data = d, weights = w, method = "GCV.Cp"))
bm("ubre", bam(y ~ s(x1) + s(x2), data = d, scale = 0.1, method = "GCV.Cp"))
bm("id", bam(y ~ s(x1, id = 1) + s(x2, id = 1), data = d, method = "GCV.Cp"))
bm("pfix", bam(y ~ s(x1) + s(x2) + s(x3), data = d, sp = c(-1, 0.5, -1), method = "GCV.Cp"))
bm("afix", bam(y ~ s(x1) + s(x2), data = d, sp = c(0.1, 2), method = "GCV.Cp"))
bm("pois", bam(yp ~ s(x1) + s(x2), data = d, family = poisson(), method = "GCV.Cp"))
bm("pois_off", bam(yp ~ s(x1) + s(x2) + offset(off), data = d, family = poisson(),
                   method = "GCV.Cp"))
bm("pois_gcv", bam(yp ~ s(x1) + s(x2), data = d, family = poisson(), scale = -1,
                   method = "GCV.Cp"))

## glm(y ~ offset(off)) null deviances of gam fits outside am.fit, and bam's
## id-linked fREML coefficients (pooled sweep-and-drop constraint)
nd <- c(gam(yp ~ s(x1) + offset(off), data = d, family = poisson(),
            method = "REML")$null.deviance,
        gam(y ~ s(x1) + offset(off), data = d, method = "REML")$null.deviance)
dumpl("null_deviance.txt", list(null_deviance = nd))
dumpl("bam_id_freml.txt", list(
  coef = coef(bam(y ~ s(x1, id = 1) + s(x2, id = 1), data = d, method = "fREML"))))
