# Live-R oracle for the hea._rs (Rust nmath) parity gate.
#
# Reads a spec + raw-f64 input files written by tests/conftest.py
# (run_rs_r_oracle), evaluates R's d/p/q with hea's parameterisation, and writes
# raw-f64 outputs. The Rust kernels and R both evaluate through the platform's
# scalar libm, so comparing them ON THE SAME MACHINE is bit-exact (0-ulp) on
# every platform — which committed cross-platform pins could not be (glibc-R
# differs from Apple-libm-R at the last ulp). Inputs travel as raw f64 so they
# are bit-identical to the arrays the test feeds the Rust side.
#
# spec.txt: one case per line, "name|fn|arg0.bin,arg1.bin,...|FLAG,FLAG"
# (the flag field is empty for no-flag kernels). Outputs: "<name>.out.bin".

args <- commandArgs(trailingOnly = TRUE)
td <- args[1]

rd <- function(n) readBin(file.path(td, n), "double", n = 1e9, size = 8, endian = "little")
wr <- function(n, v) writeBin(as.double(v), file.path(td, n), size = 8, endian = "little")

# Wrappers take args in hea's order/parameterisation, flags last, and translate
# to R's call. hea (like nmath) parameterises the exponential by SCALE; R's
# d/p/qexp take RATE = 1/scale.
hea_pnorm  <- function(x, mu, sigma, lt, lp) pnorm(x, mu, sigma, lower.tail = lt, log.p = lp)
hea_qnorm  <- function(p, mu, sigma, lt, lp) qnorm(p, mu, sigma, lower.tail = lt, log.p = lp)
hea_dnorm  <- function(x, mu, sigma, gl)     dnorm(x, mu, sigma, log = gl)
hea_pgamma <- function(x, shape, scale, lt, lp) pgamma(x, shape, scale = scale, lower.tail = lt, log.p = lp)
hea_dgamma <- function(x, shape, scale, gl)     dgamma(x, shape, scale = scale, log = gl)
hea_qgamma <- function(p, shape, scale, lt, lp) qgamma(p, shape, scale = scale, lower.tail = lt, log.p = lp)
hea_pbeta  <- function(x, a, b, lt, lp) pbeta(x, a, b, lower.tail = lt, log.p = lp)
hea_qbeta  <- function(p, a, b, lt, lp) qbeta(p, a, b, lower.tail = lt, log.p = lp)
hea_lbeta  <- function(a, b) lbeta(a, b)
hea_pt     <- function(x, df, lt, lp) pt(x, df, lower.tail = lt, log.p = lp)
hea_dt     <- function(x, df, gl)     dt(x, df, log = gl)
hea_qt     <- function(p, df, lt, lp) qt(p, df, lower.tail = lt, log.p = lp)
hea_pf     <- function(x, d1, d2, lt, lp) pf(x, d1, d2, lower.tail = lt, log.p = lp)
hea_qf     <- function(p, d1, d2, lt, lp) qf(p, d1, d2, lower.tail = lt, log.p = lp)
hea_ppois  <- function(x, lambda, lt, lp) ppois(x, lambda, lower.tail = lt, log.p = lp)
hea_qpois  <- function(p, lambda, lt, lp) qpois(p, lambda, lower.tail = lt, log.p = lp)
hea_pbinom <- function(x, n, p, lt, lp) pbinom(x, n, p, lower.tail = lt, log.p = lp)
hea_qbinom <- function(p, n, pr, lt, lp) qbinom(p, n, pr, lower.tail = lt, log.p = lp)
hea_dpois  <- function(x, lambda, gl) dpois(x, lambda, log = gl)
hea_dbinom <- function(x, n, p, gl)   dbinom(x, n, p, log = gl)
hea_dbeta  <- function(x, a, b, gl)   dbeta(x, a, b, log = gl)
hea_dexp   <- function(x, scale, gl)     dexp(x, rate = 1 / scale, log = gl)
hea_pexp   <- function(x, scale, lt, lp) pexp(x, rate = 1 / scale, lower.tail = lt, log.p = lp)
hea_qexp   <- function(p, scale, lt, lp) qexp(p, rate = 1 / scale, lower.tail = lt, log.p = lp)
hea_lgammafn <- function(x) lgamma(x)
hea_gammafn  <- function(x) gamma(x)
hea_dcauchy  <- function(x, location, scale, gl)     dcauchy(x, location, scale, log = gl)
hea_pcauchy  <- function(x, location, scale, lt, lp) pcauchy(x, location, scale, lower.tail = lt, log.p = lp)
hea_qcauchy  <- function(p, location, scale, lt, lp) qcauchy(p, location, scale, lower.tail = lt, log.p = lp)
hea_dlogis   <- function(x, location, scale, gl)     dlogis(x, location, scale, log = gl)
hea_plogis   <- function(x, location, scale, lt, lp) plogis(x, location, scale, lower.tail = lt, log.p = lp)
hea_qlogis   <- function(p, location, scale, lt, lp) qlogis(p, location, scale, lower.tail = lt, log.p = lp)
hea_dlnorm   <- function(x, meanlog, sdlog, gl)      dlnorm(x, meanlog, sdlog, log = gl)
hea_plnorm   <- function(x, meanlog, sdlog, lt, lp)  plnorm(x, meanlog, sdlog, lower.tail = lt, log.p = lp)
hea_qlnorm   <- function(p, meanlog, sdlog, lt, lp)  qlnorm(p, meanlog, sdlog, lower.tail = lt, log.p = lp)
hea_dweibull <- function(x, shape, scale, gl)        dweibull(x, shape, scale, log = gl)
hea_pweibull <- function(x, shape, scale, lt, lp)    pweibull(x, shape, scale, lower.tail = lt, log.p = lp)
hea_qweibull <- function(p, shape, scale, lt, lp)    qweibull(p, shape, scale, lower.tail = lt, log.p = lp)
hea_dgeom    <- function(x, p, gl)       dgeom(x, p, log = gl)
hea_pgeom    <- function(x, p, lt, lp)   pgeom(x, p, lower.tail = lt, log.p = lp)
hea_qgeom    <- function(p, prob, lt, lp) qgeom(p, prob, lower.tail = lt, log.p = lp)
hea_dnbinom  <- function(x, size, prob, gl)     dnbinom(x, size, prob, log = gl)
hea_dnbinom_mu <- function(x, size, mu, gl)     dnbinom(x, size, mu = mu, log = gl)
hea_pnbinom  <- function(x, size, prob, lt, lp) pnbinom(x, size, prob, lower.tail = lt, log.p = lp)
hea_qnbinom  <- function(p, size, prob, lt, lp) qnbinom(p, size, prob, lower.tail = lt, log.p = lp)
hea_qhyper   <- function(p, m, n, k, lt, lp)    qhyper(p, m, n, k, lower.tail = lt, log.p = lp)

spec <- readLines(file.path(td, "spec.txt"))
for (line in spec) {
  if (!nzchar(line)) next
  f <- strsplit(line, "|", fixed = TRUE)[[1]]
  name <- f[1]
  fn <- f[2]
  arglist <- lapply(strsplit(f[3], ",", fixed = TRUE)[[1]], rd)
  flaglist <- list()
  if (length(f) >= 4 && !is.na(f[4]) && nzchar(f[4])) {
    flaglist <- as.list(as.logical(strsplit(f[4], ",", fixed = TRUE)[[1]]))
  }
  res <- do.call(get(paste0("hea_", fn)), c(arglist, flaglist))
  wr(paste0(name, ".out.bin"), res)
}
