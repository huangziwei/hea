## mgcv's cr, cs and cc bases (smoothCon without constraints or penalty
## scaling) and their prediction matrices, for tests/test_smooths.py. The
## covariate is rounded so that it has ties and knots fall on data values;
## the prediction points include both knot ends and points outside them.
## Run from the repository root:
##
##     Rscript tests/scripts/make_crspl_fixtures.R
suppressPackageStartupMessages(library(mgcv))
out <- "tests/fixtures/crspl"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
set.seed(11)
x <- round(runif(60), 2)
xp <- c(-0.3, min(x), 0.25, 0.5, max(x), 1.4, x[1:4])
writeBin(x, file.path(out, "x.bin"))
writeBin(xp, file.path(out, "xp.bin"))
dat <- data.frame(x = x)
for (bs in c("cr", "cs", "cc")) for (k in c(4, 5, 12)) {
  sm <- smoothCon(eval(parse(text = sprintf("s(x, bs='%s', k=%d)", bs, k))), data = dat,
                  absorb.cons = FALSE, scale.penalty = FALSE)[[1]]
  tag <- sprintf("%s_%d", bs, k)
  writeBin(as.double(sm$X), file.path(out, paste0(tag, "_X.bin")))
  writeBin(as.double(sm$S[[1]]), file.path(out, paste0(tag, "_S.bin")))
  writeBin(as.double(sm$xp), file.path(out, paste0(tag, "_knots.bin")))
  writeBin(as.double(Predict.matrix(sm, data.frame(x = xp))), file.path(out, paste0(tag, "_P.bin")))
}
