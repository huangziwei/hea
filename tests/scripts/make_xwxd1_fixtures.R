## mgcv's XWXd with term selection (lt, rt): the C XWXd1 path that a general
## family's Hessian blocks take, on a discrete design with parametric,
## single-smooth, constrained tensor and by-factor terms. Run from the
## repository root:
##
##     Rscript tests/scripts/make_xwxd1_fixtures.R
##
## tests/fixtures/xwxd1/data.txt holds the data and weights; cases.txt one
## result per line, named lt_<terms>_rt_<terms> (1-based, "none" for NULL),
## dimensions and column-major values as C99 hexadecimal floats.
suppressPackageStartupMessages(library(mgcv))
stopifnot(packageVersion("mgcv") == "1.9.4")
out <- "tests/fixtures/xwxd1"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
hex <- function(v) paste(sprintf("%a", as.double(v)), collapse = " ")

set.seed(11); n <- 400
d <- data.frame(x = runif(n), z = runif(n), fac = factor(sample(c("a", "b", "c"), n, TRUE)))
d$y <- sin(2 * pi * d$x) + d$z^2 + as.numeric(d$fac) / 3 + rnorm(n) * 0.3
w <- runif(n, -0.5, 2)
G <- bam(y ~ fac + s(z) + te(x, z) + s(x, by = fac), data = d, discrete = TRUE, fit = FALSE)
nt <- length(G$ts)
width <- sapply(seq_len(nt), function(i) {
  q <- G$ts[i]:(G$ts[i] + G$dt[i] - 1)
  prod(sapply(G$Xd[q], ncol)) - (G$qc[i] > 0)
})
Xd <- G$Xd
attr(Xd, "lpip") <- split(seq_len(sum(width)), rep(seq_len(nt), width))

con <- file(file.path(out, "data.txt"), "w")
writeLines(paste("x", n, hex(d$x)), con)
writeLines(paste("z", n, hex(d$z)), con)
writeLines(paste("fac", n, hex(as.integer(d$fac))), con)
writeLines(paste("y", n, hex(d$y)), con)
writeLines(paste("w", n, hex(w)), con)
writeLines(paste("width", nt, hex(width)), con)
close(con)

cases <- list(list(NULL, NULL), list(c(1, 2), NULL), list(NULL, c(2, 3)),
              list(1:3, 1:3), list(1:7, 1:7), list(c(1, 3), c(2, 4)),
              list(4, c(1, 2)), list(c(1, 4), c(4, 5, 6)), list(4, 1:7))
con <- file(file.path(out, "cases.txt"), "w")
for (cs in cases) {
  lt <- cs[[1]]; rt <- cs[[2]]
  r <- XWXd(Xd, w, G$kd, G$ks, G$ts, G$dt, G$v, G$qc, lt = lt, rt = rt)
  tag <- function(v) if (is.null(v)) "none" else paste(v, collapse = "-")
  writeLines(paste(sprintf("lt_%s_rt_%s", tag(lt), tag(rt)),
                   paste(dim(r), collapse = ","), hex(r)), con)
}
close(con)
cat(nt, "terms, widths", width, "\n")
