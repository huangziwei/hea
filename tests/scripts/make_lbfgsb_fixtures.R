## R's optim(method = "L-BFGS-B") on small smooth problems of 3 and 4
## parameters, where its 2*col-term dot products reach five terms and
## Accelerate's ddot decides their rounding, for tests/test_optim_fixtures.py.
## Run from the repository root:
##
##     Rscript tests/scripts/make_lbfgsb_fixtures.R
##
## tests/fixtures/lbfgsb/<case>.txt.gz holds every point optim evaluated the
## objective at (rows of "xs"), its par, value and counts. One array per
## line: its name, dimensions joined by commas and the values as C99
## hexadecimal floats (sprintf("%a")), matrices column-major. The objective
## and gradient are written with explicit products so that the test's Python
## copies evaluate the same doubles.
out <- "tests/fixtures/lbfgsb"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
dumpl <- function(file, lst) {
  con <- gzfile(file.path(out, file), "w")
  for (k in names(lst)) {
    v <- lst[[k]]
    d <- dim(v); if (is.null(d)) d <- length(v)
    writeLines(paste(k, paste(d, collapse = ","),
                     paste(sprintf("%a", as.double(v)), collapse = " ")), con)
  }
  close(con)
}
xs <- NULL
fn3 <- function(x) {
  xs <<- rbind(xs, x)
  a <- x[1] - 1; b <- x[2] + 2; c3 <- x[3] - 3; e <- x[1] * x[2] - 1
  ((((a * a) * (a * a) + (b * b) * (b * b)) + (c3 * c3) * (c3 * c3)) + e * e) + 0.5 * x[3]
}
gr3 <- function(x) {
  a <- x[1] - 1; b <- x[2] + 2; c3 <- x[3] - 3; e <- x[1] * x[2] - 1
  c(4 * ((a * a) * a) + 2 * (e * x[2]), 4 * ((b * b) * b) + 2 * (e * x[1]),
    4 * ((c3 * c3) * c3) + 0.5)
}
fn4 <- function(x) {
  xs <<- rbind(xs, x)
  s <- 0
  for (i in 1:3) {
    t <- x[i + 1] - x[i] * x[i]; u <- 1 - x[i]
    s <- s + (100 * (t * t) + u * u)
  }
  s
}
gr4 <- function(x) {
  g <- c(0, 0, 0, 0)
  for (i in 1:3) {
    t <- x[i + 1] - x[i] * x[i]
    g[i] <- g[i] + (-400 * x[i] * t - 2 * (1 - x[i]))
    g[i + 1] <- g[i + 1] + 200 * t
  }
  g
}
cases <- list(
  quartic3_m3 = list(fn3, gr3, c(0.3, 0.2, -0.5), 3, -Inf, Inf),
  quartic3_m5 = list(fn3, gr3, c(0.3, 0.2, -0.5), 5, -Inf, Inf),
  rosen4_m5 = list(fn4, gr4, c(-1.2, 1, -0.5, 0.8), 5, -Inf, Inf),
  rosen4_box = list(fn4, gr4, c(-1.2, 1, -0.5, 0.8), 4, c(-1.5, -1, -2, 0), c(2, 2, 1.5, 3))
)
for (case in names(cases)) {
  cs <- cases[[case]]
  xs <- NULL
  r <- optim(cs[[3]], cs[[1]], cs[[2]], method = "L-BFGS-B", lower = cs[[5]], upper = cs[[6]],
             control = list(lmm = cs[[4]], fnscale = 2.5, factr = 1e7))
  dumpl(paste0(case, ".txt.gz"), list(xs = unname(xs), par = r$par, value = r$value,
                                      counts = as.double(r$counts)))
}
cat(length(cases), "optim runs\n")
