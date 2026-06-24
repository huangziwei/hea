## Non-discrete binomial bam parity oracle (a near-separation stress fit).
##
## A binomial near-separation fit (steep η = 14·sin(2πx), k=20) is a stress test
## for the non-discrete generalized PIRLS path (mgcv ``bgam.fit``, bam.r:909-1353)
## on a scale-known canonical-link family — complementing the scale-unknown
## (estscale gamma/tweedie) and extended-family (scat) non-discrete oracles.
## Data is generated with R's own RNG (set.seed + runif + rbinom) — the project
## never uses numpy RNG.
##
## Dumps (tests/fixtures/bam_nondiscrete_binomial/): data.csv, fitted.csv, meta.csv.

suppressMessages(library(mgcv))
root <- "tests/fixtures/bam_nondiscrete_binomial"
dir.create(root, showWarnings = FALSE, recursive = TRUE)

set.seed(0)
n <- 200
x <- sort(runif(n))
eta <- 14 * sin(2 * pi * x)        # steep ⇒ near-separation ⇒ hard binomial fit
y <- rbinom(n, 1, plogis(eta))
df <- data.frame(y = y, x = x)

m <- bam(y ~ s(x, k = 20), data = df, family = binomial(),
         method = "fREML", discrete = FALSE)

write.csv(df, file.path(root, "data.csv"), row.names = FALSE)
writeLines(format(fitted(m), digits = 17), file.path(root, "fitted.csv"))
meta <- data.frame(sp = m$sp[1], edf_total = sum(m$edf), iter = m$iter,
                   n = nrow(df))
write.csv(meta, file.path(root, "meta.csv"), row.names = FALSE)
cat(sprintf("nd-binomial: iter=%d sp=%.8g edf=%.6f\n",
            m$iter, m$sp[1], sum(m$edf)))
