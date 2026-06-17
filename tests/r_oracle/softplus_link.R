## Softplus link for R (stats::glm + mgcv::gam) — reference implementation.
## ===========================================================================
## μ = softplus(η) = log(1 + e^η);  g(μ) = log(e^μ − 1), μ > 0;
## g'(μ) = 1/(1 − e^{−μ}).
##
## This is the R twin of hea's ``SoftplusLink`` (hea/family.py) — the comp-neuro
## soft-rectifier *mean* link for Poisson point-process / RF GLMs (Paninski
## 2004; Pillow et al.): log-link-like (μ≈e^η) at low rates, identity-like
## (μ≈η) at high rates, so no exponential blow-up. It is NOT an mgcv/make.link
## built-in, so this file is kept so we can:
##   * regenerate the hea softplus test oracles
##     (tests/test_family.py::test_softplus_poisson_glm_matches_mgcv,
##      tests/test_gam.py::test_softplus_poisson_gam_matches_mgcv), and
##   * port the ALD adaptive-mass smooths (plan gam-rf-estimation-softplus-ald)
##     back to mgcv/R later, where this link is the natural Poisson nonlinearity.
##
## Derivatives, with u = e^{−μ}, s = 1 − u (use -expm1(-μ) for s at small μ):
##   g''   = −u/s²            g2g = g''/g'²   = −u
##   g'''  =  u(1+u)/s³       g3g = g'''/g'³  =  u(1+u)
##   g'''' = −u(1+4u+u²)/s⁴   g4g = g''''/g'⁴ = −u(1+4u+u²)
## ===========================================================================

## ---- link-glm object (sufficient for stats::glm) --------------------------
softplus_link <- function() {
  structure(list(
    linkfun  = function(mu)  log(expm1(mu)),       # η = log(e^μ − 1)
    linkinv  = function(eta) log1p(exp(eta)),      # μ = log(1 + e^η)
    mu.eta   = function(eta) 1 / (1 + exp(-eta)),  # dμ/dη = σ(η)
    valideta = function(eta) TRUE,                 # μ > 0 ∀ finite η
    name = "softplus"
  ), class = "link-glm")
}

## ---- attach to a family for mgcv::gam -------------------------------------
## mgcv's fix.family.link.family returns the family unchanged when it already
## carries d2link/d3link/d4link (gam.fit3.r), so supplying the analytic link
## derivatives lets gam() accept this non-built-in link. g2g/g3g/g4g are added
## as well so it also works under EXTENDED families (scat/nb/tw/…, gamlss),
## where fix.family.link.extended.family consumes them.
softplus_family <- function(base = poisson()) {
  lk <- softplus_link()
  base$link     <- "softplus"
  base$linkfun  <- lk$linkfun
  base$linkinv  <- lk$linkinv
  base$mu.eta   <- lk$mu.eta
  base$valideta <- lk$valideta
  base$canonical <- "none"          # force the non-canonical full-Newton path
  u <- function(mu) exp(-mu)
  s <- function(mu) -expm1(-mu)      # 1 − e^{−μ}, accurate as μ → 0⁺
  base$d2link <- function(mu) -u(mu) / s(mu)^2
  base$d3link <- function(mu)  u(mu) * (1 + u(mu)) / s(mu)^3
  base$d4link <- function(mu) -u(mu) * (1 + 4 * u(mu) + u(mu)^2) / s(mu)^4
  base$g2g    <- function(mu) -u(mu)
  base$g3g    <- function(mu)  u(mu) * (1 + u(mu))
  base$g4g    <- function(mu) -u(mu) * (1 + 4 * u(mu) + u(mu)^2)
  base
}

## ---- usage -----------------------------------------------------------------
## glm:  glm(y ~ x + z, family = poisson(link = softplus_link()))
## gam:  mgcv::gam(y ~ s(x), family = softplus_family(poisson()), method = "REML")
##
## The hea test fixtures draw their data through hea.R.rng (RMersenneTwister),
## which is bit-exact to R's RNG — runif AND rpois (R's rejection samplers are
## fully ported + 3-way pinned, tests/test_rs_rng_parity.py). So the same
## set.seed(k) + draw order reproduces the test data here exactly; the block
## below regenerates the two inline pins.

## ---- standalone oracle / regenerator (Rscript tests/r_oracle/softplus_link.R)
if (sys.nframe() == 0L) {
  L <- softplus_link()
  stopifnot(max(abs(L$linkfun(L$linkinv(seq(-9, 9, 0.5))) - seq(-9, 9, 0.5))) < 1e-9)

  ## --- pin 1: test_softplus_poisson_glm_matches_mgcv (RGenerator(7) order) --
  set.seed(7)
  n <- 300; x <- runif(n, -1, 2); z <- runif(n, -1, 1)
  y <- rpois(n, L$linkinv(0.6 + 1.1 * x - 0.5 * z))
  g <- glm(y ~ x + z, family = poisson(link = softplus_link()))
  cat("glm pin  coef:", sprintf("%.9g", coef(g)),
      " dev:", sprintf("%.9g", deviance(g)),
      "\n  expect: 0.595080256 1.21651587 -0.704083015  dev 328.072958\n")

  ## --- pin 2: test_softplus_poisson_gam_matches_mgcv (RGenerator(3) order) --
  if (requireNamespace("mgcv", quietly = TRUE)) {
    set.seed(3)
    n <- 200; x <- runif(n, 0, 1)
    y <- rpois(n, L$linkinv(1.0 + 1.5 * sin(2 * pi * x)))
    m <- mgcv::gam(y ~ s(x), family = softplus_family(poisson()), method = "REML")
    cat("gam pin  edf:", sprintf("%.7g", sum(m$edf)),
        " REML/2:", sprintf("%.7g", m$gcv.ubre),
        " dev:", sprintf("%.7g", deviance(m)),
        " coef0:", sprintf("%.7g", coef(m)[1]),
        "\n  expect: edf 5.718371  REML/2 306.8628  dev 231.49  coef0 1.21296\n")
  }
  cat("softplus_link.R OK\n")
}
