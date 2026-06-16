#!/usr/bin/env Rscript
# Live-R oracle for the RNG 3-way parity gate (tests/test_rs_rng_parity.py).
#
# Reads `spec.txt` in the work dir: line 1 is the integer seed; each remaining
# line is `name|rcall|nparams`. Per case, any `name__<i>.bin` files (raw little-
# endian f64) are loaded into R variables `p0`, `p1`, ... ; then R runs
# `set.seed(seed)` and evaluates `rcall` (which may reference p0/p1/...), writing
# the resulting numeric vector to `name.out.bin`. R's default RNGkind is assumed
# (Mersenne-Twister / Inversion / Rejection — R >= 3.6), which is what hea ports.
args <- commandArgs(trailingOnly = TRUE)
workdir <- args[1]
spec <- readLines(file.path(workdir, "spec.txt"))
seed <- as.integer(spec[1])

for (line in spec[-1]) {
    if (!nzchar(line)) next
    parts <- strsplit(line, "|", fixed = TRUE)[[1]]
    name <- parts[1]
    rcall <- parts[2]
    np <- as.integer(parts[3])
    if (!is.na(np) && np > 0) {
        for (i in 0:(np - 1)) {
            fn <- file.path(workdir, paste0(name, "__", i, ".bin"))
            assign(paste0("p", i), readBin(fn, "double", n = 1e9, size = 8))
        }
    }
    set.seed(seed)
    v <- eval(parse(text = rcall))
    writeBin(as.double(v), file.path(workdir, paste0(name, ".out.bin")), size = 8)
}
