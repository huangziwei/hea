## Non-discrete extended-family (scat) bam oracle.
##
## ``bam(family=scat, discrete=FALSE)`` routes through mgcv ``bgam.fit``
## (bam.r:909-1353), NOT ``bgam.fitd``. The two differ in their PIRLS cadence:
## bgam.fit estimates the family theta at the END of each iteration
## (bam.r:1204), so the next iteration's working-model build uses the PREVIOUS
## iteration's theta; bgam.fitd estimates theta mid-iteration (before the build,
## bam.r:615) so the build uses this iteration's theta. The hea shared loop was
## bgam.fitd-shaped, diverging ~3e-6 on the fitted values for this exact fit
## (iter 10 vs mgcv 12). This oracle pins the faithful bgam.fit cadence.
##
## Uses the SAME data as the discrete ``simple`` scat oracle (seed 7, n=250),
## fit with discrete=FALSE. Dumps (tests/fixtures/scat_bam_nondiscrete/simple/):
##   data.csv, sp.csv, theta.csv, edf.csv, fitted.csv.

suppressMessages(library(mgcv))

.dump <- function(m, df, sub) {
  root <- file.path("tests/fixtures/scat_bam_nondiscrete", sub)
  dir.create(root, showWarnings = FALSE, recursive = TRUE)
  write.csv(df, file.path(root, "data.csv"), row.names = FALSE)
  writeLines(format(m$sp, digits = 17), file.path(root, "sp.csv"))
  writeLines(format(m$family$getTheta(TRUE), digits = 17),
             file.path(root, "theta.csv"))
  writeLines(format(sum(m$edf), digits = 17), file.path(root, "edf.csv"))
  writeLines(format(fitted(m), digits = 17), file.path(root, "fitted.csv"))
  cat(sprintf("scat-nd %-7s: iter=%d sp=%s edf=%.6f theta=(%.6f,%.6f)\n",
              sub, m$iter, paste(format(m$sp, digits = 4), collapse = ","),
              sum(m$edf),
              m$family$getTheta(TRUE)[1], m$family$getTheta(TRUE)[2]))
}

# ---------- simple : single smooth -----------------------------------------
set.seed(7)
n <- 250
x <- runif(n)
y <- 2 * sin(2 * pi * x) + rt(n, df = 5) * 0.3
df1 <- data.frame(y = y, x = x)
.dump(bam(y ~ s(x, k = 10), data = df1, family = scat(min.df = 5),
          method = "fREML", discrete = FALSE), df1, "simple")

# ---------- factor : 3-level by=factor smooth ------------------------------
set.seed(11)
n <- 300
x <- runif(n)
g <- factor(rep(c("a", "b", "c"), length.out = n))
mu <- 2 * sin(2 * pi * x) + as.numeric(g) * 0.3
y <- mu + rt(n, df = 5) * 0.25
df2 <- data.frame(y = y, x = x, g = g)
.dump(bam(y ~ g + s(x, by = g, k = 10), data = df2, family = scat(min.df = 5),
          method = "fREML", discrete = FALSE), df2, "factor")
