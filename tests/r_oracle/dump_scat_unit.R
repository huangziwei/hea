## Generates the per-output unit-level oracle for ``hea.family.Scat``.
##
## Pinned outputs (under tests/fixtures/scat_unit/):
##   inputs.csv     — (y, mu, wt) probe of length n
##   theta.csv      — log-internal θ = (log(ν - min_df), log σ)
##   dd_lvl{0,1,2}.csv — column-wise dump of mgcv ``scat$Dd`` at each level
##   ls_summary.csv — (ls, lsth1[0], lsth1[1], lsth2[1,1])
##   LSTH1.csv      — per-obs first-derivative matrix from ``ls``
##   dev.csv        — ``scat$dev.resids(y, mu, wt, theta)``
##   aic.csv        — ``scat$aic(y, mu, theta, wt, dev=NULL)``
##   preinit.csv    — initial Theta from ``scat$preinitialize(y, family)``
##
## min.df is fixed at 5 so the test exercises the "non-default min_df"
## path; theta is chosen to keep ν away from min.df (to avoid degenerate
## branches where the formulas have spurious zeros that hide bugs).
##
## Re-run this script after upgrading mgcv to refresh the oracle.

suppressMessages(library(mgcv))

set.seed(42)
n <- 30
y <- rnorm(n) + sin(seq_len(n) / 3)
mu <- y + rnorm(n, sd = 0.3)
wt <- rep(1.0, n)
theta <- c(0.4, -0.3)
min_df <- 5

fam <- scat(min.df = min_df)
fam$putTheta(theta)

dd0 <- fam$Dd(y, mu, theta, wt, level = 0)
dd1 <- fam$Dd(y, mu, theta, wt, level = 1)
dd2 <- fam$Dd(y, mu, theta, wt, level = 2)

dir <- "tests/fixtures/scat_unit"
dir.create(dir, showWarnings = FALSE, recursive = TRUE)

write.csv(data.frame(y = y, mu = mu, wt = wt),
          file.path(dir, "inputs.csv"), row.names = FALSE)
writeLines(format(theta, digits = 15), file.path(dir, "theta.csv"))

write.table(cbind(dd0$Dmu, dd0$Dmu2, dd0$EDmu2),
            file.path(dir, "dd_lvl0.csv"),
            sep = ",", row.names = FALSE,
            col.names = c("Dmu", "Dmu2", "EDmu2"))
write.table(cbind(dd1$Dth, dd1$Dmuth, dd1$Dmu2th, dd1$EDmu2th,
                  dd1$Dmu3, dd1$EDmu3),
            file.path(dir, "dd_lvl1.csv"),
            sep = ",", row.names = FALSE,
            col.names = c("Dth0", "Dth1", "Dmuth0", "Dmuth1",
                          "Dmu2th0", "Dmu2th1", "EDmu2th0", "EDmu2th1",
                          "Dmu3", "EDmu3"))
write.table(cbind(dd2$Dmu4, dd2$Dmu3th, dd2$Dmu2th2, dd2$Dmuth2, dd2$Dth2),
            file.path(dir, "dd_lvl2.csv"),
            sep = ",", row.names = FALSE,
            col.names = c("Dmu4", "Dmu3th0", "Dmu3th1",
                          "Dmu2th2_0", "Dmu2th2_1", "Dmu2th2_2",
                          "Dmuth2_0", "Dmuth2_1", "Dmuth2_2",
                          "Dth2_0", "Dth2_1", "Dth2_2"))

ls_out <- fam$ls(y, wt, theta, scale = 1)
write.table(ls_out$LSTH1, file.path(dir, "LSTH1.csv"),
            sep = ",", row.names = FALSE, col.names = c("col0", "col1"))
writeLines(c(format(ls_out$ls, digits = 15),
             format(ls_out$lsth1[1], digits = 15),
             format(ls_out$lsth1[2], digits = 15),
             format(ls_out$lsth2[1, 1], digits = 15)),
           file.path(dir, "ls_summary.csv"))

dev <- fam$dev.resids(y, mu, wt, theta = theta)
aic <- fam$aic(y, mu, theta, wt, dev = NULL)
write.table(dev, file.path(dir, "dev.csv"),
            sep = ",", row.names = FALSE, col.names = "dev")
writeLines(format(aic, digits = 15), file.path(dir, "aic.csv"))

fam2 <- scat(min.df = 5)
pini <- fam2$preinitialize(y, fam2)
writeLines(format(pini$Theta, digits = 15), file.path(dir, "preinit.csv"))
