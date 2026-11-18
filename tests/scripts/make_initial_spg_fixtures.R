## mgcv's initial.spg for a general family, for tests/test_gam.py: the
## family's starting coefficients and the initial log smoothing parameters
## for gam(list(accel ~ s(times, k=20, bs="ad"), ~ s(times)), gaulss()) on
## MASS::mcycle, computed as estimate.gam does (Sl.setup, Sl.initial.repara,
## totalPenaltySpace, then initial.spg on the reparameterized X with the
## original-basis penalties). Run from the repository root:
##
##     Rscript tests/scripts/make_initial_spg_fixtures.R
suppressPackageStartupMessages({library(mgcv); library(MASS)})
out <- "tests/fixtures/initial_spg"
dir.create(out, showWarnings = FALSE, recursive = TRUE)
G <- gam(list(accel ~ s(times, k = 20, bs = "ad"), ~ s(times)), family = gaulss(),
         data = mcycle, fit = FALSE)
Sl <- mgcv:::Sl.setup(G)
X2 <- mgcv:::Sl.initial.repara(Sl, G$X, both.sides = FALSE)
E <- mgcv:::totalPenaltySpace(G$S, G$H, G$off, ncol(G$X))$E
fam <- mgcv:::fix.family(G$family)
env <- new.env(parent = asNamespace("mgcv"))
env$x <- X2; env$y <- G$y; env$weights <- G$w; env$E <- E; env$offset <- G$offset
env$family <- fam; env$nobs <- nrow(X2); env$start <- NULL; env$mustart <- NULL
env$etastart <- NULL
eval(fam$initialize, env)
writeBin(as.double(env$start), file.path(out, "mcycle_gaulss_ad_start.bin"))
lsp <- log(mgcv:::initial.spg(X2, G$y, G$w, fam, G$S, G$rank, G$off, offset = G$offset,
                              L = G$L, lsp0 = G$lsp0, E = E))
writeBin(as.double(lsp), file.path(out, "mcycle_gaulss_ad_lsp0.bin"))

## initial.spg's L regression (lm(lsp ~ L - 1 + offset(lsp0))) for two
## id-linked smooths under a binomial family.
set.seed(45)
dat <- data.frame(x = runif(200), z = runif(200))
dat$yb <- as.numeric(runif(200) < 0.5 + 0.3 * sin(6 * dat$x))
G <- gam(yb ~ s(x, id = 1) + s(z, id = 1), data = dat, family = binomial, method = "REML",
         fit = FALSE)
lsp <- log(mgcv:::initial.spg(G$X, G$y, G$w, mgcv:::fix.family(G$family), G$S, G$rank, G$off,
                              offset = G$offset, L = G$L, lsp0 = G$lsp0))
writeBin(as.double(lsp), file.path(out, "binomial_id_lsp0.bin"))

## G$rank (the per-penalty ranks initial.spg and Sl.setup read) for a spread
## of smooth classes.
set.seed(7)
n <- 300
x <- runif(n); z <- runif(n); w <- runif(n); f <- factor(paste0("L", floor(runif(n) * 4)))
dat <- data.frame(x = x, z = z, w = w, f = f, y = x)
specs <- readLines("tests/fixtures/initial_spg/rank_specs.txt")
ranks <- lapply(specs, function(sp) gam(as.formula(paste("y ~", sp)), data = dat, fit = FALSE)$rank)
writeLines(vapply(ranks, function(r) paste(r, collapse = " "), ""),
           file.path(out, "rank_values.txt"))

## Each smooth's null.space.dim (summary.gam sends a smooth to reTest when it
## is 0), for the same specs without and with select = TRUE.
nsd <- function(sel) lapply(specs, function(sp) {
  G <- gam(as.formula(paste("y ~", sp)), data = dat, fit = FALSE, select = sel)
  vapply(G$smooth, function(s) as.integer(s$null.space.dim), 0L)
})
writeLines(vapply(nsd(FALSE), function(r) paste(r, collapse = " "), ""),
           file.path(out, "null_space_dim_values.txt"))
writeLines(vapply(nsd(TRUE), function(r) paste(r, collapse = " "), ""),
           file.path(out, "null_space_dim_select_values.txt"))

## The fitted mcycle gaulss model (adaptive mean smooth): coefficients, edf,
## edf1, edf2, Vp, Vc and the smoothing parameters, for the bit-exact fit.
m <- gam(list(accel ~ s(times, k = 20, bs = "ad"), ~ s(times)), family = gaulss(),
         data = mcycle, method = "REML")
writeBin(c(as.double(coef(m)), m$edf, m$edf1, m$edf2, as.double(m$Vp), as.double(m$Vc),
           log(m$sp), m$gcv.ubre), file.path(out, "mcycle_gaulss_fit.bin"))
