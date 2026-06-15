## predict.bam surface oracle for hea.models.bam.predict (plan item P4).
##
## hea's bam.predict used to reject type∉{link,response,lpmatrix} and drop
## terms=/exclude=/unconditional/iterms — even though predict.bam delegates the
## non-discrete case to predict.gam, which supports them all. These oracles pin
## the widened surface against mgcv on the gauss fixture (same data the summary
## oracle fits, where hea-bam coef matches mgcv-bam to ~1e-9).
##
## Dumps (under tests/fixtures/bam_predict/):
##   terms_fit.csv     predict(type="terms")              [z, s(x)]
##   terms_se.csv      predict(type="terms", se.fit=T)$se.fit
##   iterms_se.csv     predict(type="iterms", se.fit=T)$se.fit   (cmX-widened)
##   link_excl_sx.csv  predict(type="link", exclude="s(x)")
##   link_se_uncond.csv predict(type="link", se.fit=T, unconditional=T)$se.fit

suppressMessages(library(mgcv))

dg <- read.csv("tests/fixtures/bam_summary/gauss/data.csv")
m <- bam(y ~ z + s(x, k = 10), data = dg)          # fREML default -> Vc avail.

dir.create("tests/fixtures/bam_predict", showWarnings = FALSE, recursive = TRUE)

tf <- predict(m, type = "terms")
write.csv(tf, "tests/fixtures/bam_predict/terms_fit.csv", row.names = FALSE)

ts <- predict(m, type = "terms", se.fit = TRUE)
write.csv(ts$se.fit, "tests/fixtures/bam_predict/terms_se.csv", row.names = FALSE)

it <- predict(m, type = "iterms", se.fit = TRUE)
write.csv(it$se.fit, "tests/fixtures/bam_predict/iterms_se.csv", row.names = FALSE)

le <- predict(m, type = "link", exclude = "s(x)")
writeLines(format(as.numeric(le), digits = 15),
           "tests/fixtures/bam_predict/link_excl_sx.csv")

lu <- predict(m, type = "link", se.fit = TRUE, unconditional = TRUE)
writeLines(format(as.numeric(lu$se.fit), digits = 15),
           "tests/fixtures/bam_predict/link_se_uncond.csv")

cat("predict oracle: terms cols =", paste(colnames(tf), collapse = ", "), "\n")
