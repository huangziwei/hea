## Non-discrete bam fit that TRIGGERS the bgam.fit step-halving (bam.r:1163-1190).
##
## A binomial near-separation fit (steep η = 14·sin(2πx), k=20) overshoots the
## penalised deviance early, so mgcv's bgam.fit halves the β step (kk≥1) before
## re-accumulating the working model. hea's _bgam_fit_loop must follow the
## NON-discrete halving cadence here (β'Sβ via Sl.Sb, θ0, kk<6), NOT bgam.fitd's
## (sum(rSb²), current θ, kk<30). Reads the committed numpy-generated data.csv
## (rng default_rng(0)) and dumps the mgcv oracle.
##
## Dumps (tests/fixtures/bam_nondiscrete_halving/): fitted.csv, meta.csv.

suppressMessages(library(mgcv))
root <- "tests/fixtures/bam_nondiscrete_halving"
df <- read.csv(file.path(root, "data.csv"))
m <- bam(y ~ s(x, k = 20), data = df, family = binomial(),
         method = "fREML", discrete = FALSE)
writeLines(format(fitted(m), digits = 17), file.path(root, "fitted.csv"))
meta <- data.frame(sp = m$sp[1], edf_total = sum(m$edf), iter = m$iter,
                   n = nrow(df))
write.csv(meta, file.path(root, "meta.csv"), row.names = FALSE)
cat(sprintf("nd-halving: iter=%d sp=%.8g edf=%.6f\n",
            m$iter, m$sp[1], sum(m$edf)))
