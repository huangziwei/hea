//! The platform's BLAS, behind `--features blas`.
//!
//! ```text
//! .venv/bin/maturin develop --release --features blas
//! ```
//!
//! # Why this is the faithful path, not the unfaithful one
//!
//! Upstream's supernodal factorization is four BLAS calls
//! (`t_cholmod_super_numeric_worker.c`): one `dsyrk ("L","N")` and one
//! `dgemm ("N","C")` per descendant update, at `:769` and `:824`, and one
//! `dpotrf ("L")` and one `dtrsm ("R","L","C","N")` per supernode, at `:1023`
//! and `:1175`. Its supernodal solve is eight more
//! (`t_cholmod_super_solve_worker.c`), a `trsv`/`gemv` pair per half at
//! `nrhs == 1` and a `trsm`/`gemm` pair per half above it.
//! [`super::dense`]'s twelve entry points are those twelve calls, and this
//! module hands each straight to the vendor.
//!
//! Everything hea does *around* them — cutting a descendant's update into column
//! strips, blocking `potrf_l` and `trsm_rlt` into panels — exists because hea's
//! own kernels are one thread per call, so the parallelism had to be built above
//! them. A threaded BLAS supplies it, so this path issues upstream's call
//! sequence unsplit, which is what the port would have said all along if a BLAS
//! had been available to it.
//!
//! So the arithmetic here is not a *third* rounding source for the bit-exact
//! comparisons to work around. It is the same source on both sides: the
//! supernodal oracle is upstream's own C, and it is already built twice, once
//! against Accelerate and once against hea's kernels.
//!
//! # Which library, per platform
//!
//! | target | library | why |
//! |---|---|---|
//! | macOS | **Accelerate** | Apple's own, linked as a system framework, and the only thing on this hardware that reaches the matrix coprocessor — one core of it runs `dgemm` an order of magnitude above what the NEON ISA can issue. Nothing to ship. |
//! | Linux | **OpenBLAS** | the reference-quality open BLAS, and the one the scientific-Python wheels have shipped for a decade. |
//! | Windows | **OpenBLAS** | same, and the only realistic option: the CHOLMOD wrapper hea replaces does not build there at all. |
//!
//! On macOS the framework is present on every machine, so there is nothing to
//! find. Elsewhere the build needs a `libopenblas` to link against, and the
//! well-proven one to take is the `scipy-openblas32` distribution — that is not
//! scipy, it is a plain OpenBLAS build (`DYNAMIC_ARCH`, so it dispatches on the
//! running CPU) published as a wheel for exactly this purpose, and it is the
//! same binary the numpy and scipy wheels carry. Its symbols are prefixed
//! `scipy_`, which is why the `link_name`s below are conditional. `build.rs`
//! finds it from `HEA_BLAS_LIB_DIR` or `pkg-config`, and the wheel repair step
//! vendors it, so the installed wheel keeps needing nothing from the system —
//! which is the point of the port, and the thing the wrapper it replaces cannot
//! do without a system `libsuitesparse-dev`.
//!
//! # Choosing the backend, and testing the one you are not on
//!
//! `blas-openblas` forces the OpenBLAS arm on *any* target, macOS included.
//! That is a real preference — someone may want one BLAS across a fleet — but it
//! is here mostly so the arm that ships to Linux and Windows can be built and
//! run on a machine that is neither: the `scipy-openblas32` wheel has a macOS
//! build exporting the same prefixed symbols, so
//!
//! ```text
//! HEA_BLAS_LIB_DIR=$(python -c "import scipy_openblas32 as s; print(s.get_lib_dir())") \
//!     maturin develop --release --features blas,blas-openblas
//! ```
//!
//! exercises these exact `link_name`s, this exact `build.rs` path and this exact
//! call sequence, against the same library those platforms will load. What it
//! cannot cover is the platform's own linker and loader, and the wheel-repair
//! step that vendors the library — those need the real target, and belong in CI.
//!
//! # What it costs, and why it is off by default
//!
//! A digest taken against Accelerate is not the one OpenBLAS gives, so a pinned
//! literal stops being one number across platforms. The shape of the answer is
//! the one hea's other cross-platform pins already use — store every answer the
//! reference itself has and assert membership, never branch on the platform —
//! but changing what a pin *means* is a decision, not a side effect of a
//! performance change, so the feature ships off and the default build stays the
//! deterministic one the pins describe.

use std::os::raw::{c_char, c_double, c_int};

/// Flops below which a call keeps hea's own kernel instead of the vendor's.
///
/// A sparse factorization is mostly tiny calls: on a 102k-row system 22,685 of
/// the 35,000 descendant updates are eight columns wide and carry 3% of the
/// flops between them. Handed to the vendor they run at **10 GF/s** — that is
/// the dispatch, not the arithmetic — where hea's own kernels do the same work
/// at 60. The big calls go the other way by a wider margin, 195 GF/s against
/// hea's 60, because they reach the matrix coprocessor. So the two paths are
/// complementary rather than competing, and the crossing is what this is.
///
/// Swept in the driver on a 102k-row system, total core ms across the four
/// kernels (`0` routes everything to the vendor):
///
/// | cutoff, kflop | 0 | 10 | **100** | 1000 | 10000 |
/// |---|---|---|---|---|---|
/// | core ms | 93.4 | 93.4 | **85.7** | 90.6 | 99.7 |
/// | wall ms | 29.5 | 29.7 | 28.0 | 28.9 | 28.2 |
///
/// The wall clock is flat across all of it and only the core column moves, so
/// this buys CPU rather than latency — which is the column hea is behind on.
///
/// # The solve crosses in the same place, which was not a given
///
/// The eight solve kernels share this cutoff, and the reason is a measurement
/// rather than a convenience. Their regime is a different one — `trsv`/`gemv`
/// are level 2 and move as many bytes as they do flops, so the coprocessor has
/// nothing to offer them, while `trsm`/`gemm` widen with the right-hand sides
/// until it does — so the crossing had no reason to land anywhere near the
/// factorization's.
///
/// Swept over the benchmark corpus at `nrhs` 1, 4, 16 and 32, as the geometric
/// mean of CHOLMOD-on-Accelerate's solve time over hea's, above 1 meaning hea
/// is ahead:
///
/// | cutoff, kflop | ∞ | 4000 | 1000 | 250 | **100** | 50 | 25 | 0 |
/// |---|---|---|---|---|---|---|---|---|
/// | wall | 0.852 | 0.897 | 0.923 | 0.978 | **1.014** | 1.020 | 1.008 | 0.950 |
/// | core | 0.858 | 0.914 | 0.913 | 0.984 | **1.028** | 1.022 | 1.032 | 0.938 |
///
/// `∞` is the control — no routing, which is what the solve did before these
/// kernels were bound — and it is where the whole win is: routing at all is
/// worth **19%**, and it is the difference between trailing the reference by
/// 15% and leading it. Where inside the 25-100 kflop basin the cutoff sits is
/// not: those three columns are one flat bottom, and their spread is smaller
/// than the run-to-run spread of the same instrument.
///
/// The win is concentrated where the ceiling says it should be. Measured in
/// upstream's own C, linked once to Accelerate and once to hea's kernels so the
/// ratio is the substitution and nothing else, the vendor is behind at
/// `nrhs = 1` on the small-supernode systems (laplacian-220sq **0.86**) and
/// ahead by **1.67x** at `nrhs = 32` on the large ones. A cutoff on flops
/// separates those two without ever looking at `nrhs`, which is why one number
/// covers all eight kernels.
#[cfg(not(feature = "blas-all"))]
pub const MIN_FLOPS: f64 = 1.0e5;

/// The `0` column of that table, and the only build where this path can be
/// checked for bit-exactness.
///
/// The claim this feature exists to test is that hea on a vendor BLAS issues
/// *upstream's* call sequence — one `dsyrk` and one `dgemm` per descendant, one
/// `dpotrf` and one `dtrsm` per supernode, all full width — rather than the
/// column strips and blocked `potrf` its own kernels need. If that holds, then
/// hea linked to Accelerate and CHOLMOD linked to Accelerate are doing the same
/// arithmetic in the same order and `L->x` must agree **to the bit**.
///
/// At the shipped cutoff it cannot be tested, because the calls below the
/// crossing stay on hea's kernels and `L` is then a blend of two libraries'
/// rounding. At zero there is one library and `==` is available, and both
/// halves of the pipeline pass it over the whole SPD corpus, both stypes and
/// both orderings: **184 of 184** factorizations and **2484 of 2484** solves
/// exactly equal, no tolerance anywhere.
#[cfg(feature = "blas-all")]
pub const MIN_FLOPS: f64 = 0.0;

/// Whether a call of this many flops is worth the vendor's dispatch.
#[inline]
pub fn worth_it(flops: f64) -> bool {
    flops >= MIN_FLOPS
}

// The classic 32-bit-integer F77 BLAS and LAPACK, which both libraries export.
// The character arguments carry hidden trailing lengths in the Fortran ABI;
// every C and Rust caller omits them and every implementation ignores them,
// reading only the first byte.
#[cfg_attr(accelerate, link(name = "Accelerate", kind = "framework"))]
#[cfg_attr(not(accelerate), link(name = "scipy_openblas"))]
extern "C" {
    #[cfg_attr(not(accelerate), link_name = "scipy_dgemm_")]
    fn dgemm_(
        transa: *const c_char,
        transb: *const c_char,
        m: *const c_int,
        n: *const c_int,
        k: *const c_int,
        alpha: *const c_double,
        a: *const c_double,
        lda: *const c_int,
        b: *const c_double,
        ldb: *const c_int,
        beta: *const c_double,
        c: *mut c_double,
        ldc: *const c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_dsyrk_")]
    fn dsyrk_(
        uplo: *const c_char,
        trans: *const c_char,
        n: *const c_int,
        k: *const c_int,
        alpha: *const c_double,
        a: *const c_double,
        lda: *const c_int,
        beta: *const c_double,
        c: *mut c_double,
        ldc: *const c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_dtrsm_")]
    fn dtrsm_(
        side: *const c_char,
        uplo: *const c_char,
        transa: *const c_char,
        diag: *const c_char,
        m: *const c_int,
        n: *const c_int,
        alpha: *const c_double,
        a: *const c_double,
        lda: *const c_int,
        b: *mut c_double,
        ldb: *const c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_dpotrf_")]
    fn dpotrf_(
        uplo: *const c_char,
        n: *const c_int,
        a: *mut c_double,
        lda: *const c_int,
        info: *mut c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_dtrsv_")]
    fn dtrsv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const c_int,
        a: *const c_double,
        lda: *const c_int,
        x: *mut c_double,
        incx: *const c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_dgemv_")]
    fn dgemv_(
        trans: *const c_char,
        m: *const c_int,
        n: *const c_int,
        alpha: *const c_double,
        a: *const c_double,
        lda: *const c_int,
        x: *const c_double,
        incx: *const c_int,
        beta: *const c_double,
        y: *mut c_double,
        incy: *const c_int,
    );
}

/// OpenBLAS runs its own thread pool and, unlike Accelerate, does not stand down
/// when it is called from inside one.
///
/// hea calls these kernels from `rayon` workers, so an OpenBLAS that threads
/// underneath is nested parallelism: measured on a 102k-row system it read
/// **17 threads and 4239 core-ms** against Accelerate's 4.3 and 90.3, a 47x
/// increase in CPU for a 12x *worse* wall clock. The parallelism belongs to
/// hea's tree here — upstream leans on the BLAS for it only because its own
/// supernode loop is serial.
///
/// Called once, before any kernel. `OPENBLAS_NUM_THREADS` in the environment
/// would do the same thing and be the caller's problem to remember, which is
/// not a contract to ship.
#[cfg(not(accelerate))]
pub fn init() {
    use std::sync::Once;
    extern "C" {
        #[link_name = "scipy_openblas_set_num_threads"]
        fn openblas_set_num_threads(n: c_int);
    }
    static ONCE: Once = Once::new();
    /* SAFETY: OpenBLAS's own documented entry point for this, and `Once` makes
     * it a single call before any kernel runs. */
    ONCE.call_once(|| unsafe { openblas_set_num_threads(1) });
}

/// Accelerate needs no such call: measured at 4.3-5.4 threads inside hea's pool,
/// it already stands down.
#[cfg(accelerate)]
#[inline]
pub fn init() {}

/// A one-character F77 flag.
#[inline]
fn ch(c: u8) -> c_char {
    c as c_char
}

/// `dgemm ("N","C", m, n, k, 1, a, lda, b, ldb, 0, c, ldc)` — upstream `:824`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nt(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let (mi, ni, ki) = (m as c_int, n as c_int, k as c_int);
    let (ldai, ldbi, ldci) = (lda as c_int, ldb as c_int, ldc as c_int);
    let (one, zero) = (1.0f64, 0.0f64);
    /* SAFETY: the dimensions are the caller's own and `dense`'s debug asserts
     * bound each slice by them; `c` cannot alias `a` or `b`, being `&mut` where
     * they are `&`. */
    unsafe {
        dgemm_(
            &ch(b'N'),
            &ch(b'C'),
            &mi,
            &ni,
            &ki,
            &one,
            a.as_ptr(),
            &ldai,
            b.as_ptr(),
            &ldbi,
            &zero,
            c.as_mut_ptr(),
            &ldci,
        );
    }
}

/// `dsyrk ("L","N", n, k, 1, a, lda, 0, c, ldc)` — upstream `:769`.
pub fn syrk_ln(n: usize, k: usize, a: &[f64], lda: usize, c: &mut [f64], ldc: usize) {
    if n == 0 {
        return;
    }
    let (ni, ki) = (n as c_int, k as c_int);
    let (ldai, ldci) = (lda as c_int, ldc as c_int);
    let (one, zero) = (1.0f64, 0.0f64);
    /* SAFETY: as `gemm_nt` above. */
    unsafe {
        dsyrk_(
            &ch(b'L'),
            &ch(b'N'),
            &ni,
            &ki,
            &one,
            a.as_ptr(),
            &ldai,
            &zero,
            c.as_mut_ptr(),
            &ldci,
        );
    }
}

/// `dtrsm ("R","L","C","N", m, n, 1, a, lda, b, ldb)` — upstream `:1175`.
///
/// The worker hands one array for both operands: `a` the supernode's leading
/// `n`-by-`n` block and `b` the `m` rows below it at the same leading dimension.
pub fn trsm_rlt(m: usize, n: usize, x: &mut [f64], ld: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let (mi, ni) = (m as c_int, n as c_int);
    let ldi = ld as c_int;
    let one = 1.0f64;
    let p = x.as_mut_ptr();
    /* SAFETY: `a` is `x[..]` and `b` is `x[n..]`, which is upstream's own
     * aliasing — `dtrsm` reads the triangle and writes only the rows below it,
     * and the two do not overlap. */
    unsafe {
        dtrsm_(
            &ch(b'R'),
            &ch(b'L'),
            &ch(b'C'),
            &ch(b'N'),
            &mi,
            &ni,
            &one,
            p,
            &ldi,
            p.add(n),
            &ldi,
        );
    }
}

/// `dpotrf ("L", n, a, lda, info)` — upstream `:1023`.
///
/// Returns `INFO`, which [`super::dense::potrf_l`] reports as the 1-based column
/// that was not positive definite.
pub fn potrf_l(n: usize, a: &mut [f64], lda: usize) -> i64 {
    if n == 0 {
        return 0;
    }
    let (ni, ldai) = (n as c_int, lda as c_int);
    let mut info: c_int = 0;
    /* SAFETY: `a` is the caller's `n`-by-`n` block at `lda`, bounded by
     * `dense`'s debug assert. */
    unsafe {
        dpotrf_(&ch(b'L'), &ni, a.as_mut_ptr(), &ldai, &mut info);
    }
    info as i64
}

/* ========================================================================= */
/* === the solve, `t_cholmod_super_solve_worker.c` ========================= */
/* ========================================================================= */

/* Each of the eight below is one call site of that worker, with the same flags
 * and the same `alpha`/`beta` it fixes there. `L` is only ever read, so `a` is
 * shared and the destination is the exclusive `&mut` — none of these can alias,
 * which is what makes the raw pointers sound. */

/// `dtrsv ("L","N","N", n, a, lda, x, 1)` — solve worker `:93`.
pub fn trsv_ln(n: usize, a: &[f64], lda: usize, x: &mut [f64]) {
    if n == 0 {
        return;
    }
    let (ni, ldai, inc) = (n as c_int, lda as c_int, 1 as c_int);
    /* SAFETY: `a` is `n`-by-`n` at `lda` and `x` is `n` long, both bounded by
     * `dense`'s debug asserts; they are separate arrays. */
    unsafe {
        dtrsv_(
            &ch(b'L'),
            &ch(b'N'),
            &ch(b'N'),
            &ni,
            a.as_ptr(),
            &ldai,
            x.as_mut_ptr(),
            &inc,
        );
    }
}

/// `dtrsv ("L","C","N", n, a, lda, x, 1)` — solve worker `:398`.
pub fn trsv_lt(n: usize, a: &[f64], lda: usize, x: &mut [f64]) {
    if n == 0 {
        return;
    }
    let (ni, ldai, inc) = (n as c_int, lda as c_int, 1 as c_int);
    /* SAFETY: as `trsv_ln` above. */
    unsafe {
        dtrsv_(
            &ch(b'L'),
            &ch(b'C'),
            &ch(b'N'),
            &ni,
            a.as_ptr(),
            &ldai,
            x.as_mut_ptr(),
            &inc,
        );
    }
}

/// `dgemv ("N", m, n, -1, a, lda, x, 1, 1, y, 1)` — solve worker `:99`.
pub fn gemv_n(m: usize, n: usize, a: &[f64], lda: usize, x: &[f64], y: &mut [f64]) {
    if m == 0 || n == 0 {
        return;
    }
    let (mi, ni, ldai, inc) = (m as c_int, n as c_int, lda as c_int, 1 as c_int);
    let (minus_one, one) = (-1.0f64, 1.0f64);
    /* SAFETY: `a` is `m`-by-`n` at `lda`, `x` is `n` long and `y` is `m`, all
     * bounded by `dense`'s debug asserts; `y` is `&mut` where the others are
     * shared, so it aliases neither. */
    unsafe {
        dgemv_(
            &ch(b'N'),
            &mi,
            &ni,
            &minus_one,
            a.as_ptr(),
            &ldai,
            x.as_ptr(),
            &inc,
            &one,
            y.as_mut_ptr(),
            &inc,
        );
    }
}

/// `dgemv ("C", m, n, -1, a, lda, x, 1, 1, y, 1)` — solve worker `:388`.
///
/// The transpose swaps the two extents against [`gemv_n`]: `x` is `m` long and
/// `y` is `n`, while `a` is still `m`-by-`n` at `lda`.
pub fn gemv_t(m: usize, n: usize, a: &[f64], lda: usize, x: &[f64], y: &mut [f64]) {
    if m == 0 || n == 0 {
        return;
    }
    let (mi, ni, ldai, inc) = (m as c_int, n as c_int, lda as c_int, 1 as c_int);
    let (minus_one, one) = (-1.0f64, 1.0f64);
    /* SAFETY: as `gemv_n` above, with `x` and `y` the other way round. */
    unsafe {
        dgemv_(
            &ch(b'C'),
            &mi,
            &ni,
            &minus_one,
            a.as_ptr(),
            &ldai,
            x.as_ptr(),
            &inc,
            &one,
            y.as_mut_ptr(),
            &inc,
        );
    }
}

/// `dtrsm ("L","L","N","N", m, n, 1, a, lda, b, ldb)` — solve worker `:204`.
pub fn trsm_lln(m: usize, n: usize, a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let (mi, ni) = (m as c_int, n as c_int);
    let (ldai, ldbi) = (lda as c_int, ldb as c_int);
    let one = 1.0f64;
    /* SAFETY: `a` is `m`-by-`m` at `lda` and `b` is `m`-by-`n` at `ldb`, both
     * bounded by `dense`'s debug asserts. Unlike `trsm_rlt`, which the numeric
     * worker hands one array split by row, these are the factor and the
     * right-hand side — separate allocations. */
    unsafe {
        dtrsm_(
            &ch(b'L'),
            &ch(b'L'),
            &ch(b'N'),
            &ch(b'N'),
            &mi,
            &ni,
            &one,
            a.as_ptr(),
            &ldai,
            b.as_mut_ptr(),
            &ldbi,
        );
    }
}

/// `dtrsm ("L","L","C","N", m, n, 1, a, lda, b, ldb)` — solve worker `:505`.
pub fn trsm_llt(m: usize, n: usize, a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {
    if m == 0 || n == 0 {
        return;
    }
    let (mi, ni) = (m as c_int, n as c_int);
    let (ldai, ldbi) = (lda as c_int, ldb as c_int);
    let one = 1.0f64;
    /* SAFETY: as `trsm_lln` above. */
    unsafe {
        dtrsm_(
            &ch(b'L'),
            &ch(b'L'),
            &ch(b'C'),
            &ch(b'N'),
            &mi,
            &ni,
            &one,
            a.as_ptr(),
            &ldai,
            b.as_mut_ptr(),
            &ldbi,
        );
    }
}

/// `dgemm ("N","N", m, n, k, -1, a, lda, b, ldb, 1, c, ldc)` — solve worker
/// `:213`.
///
/// `beta` is 1 here, not 0 as in [`gemm_nt`]: the solve accumulates into `E`
/// rather than overwriting it.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nn(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let (mi, ni, ki) = (m as c_int, n as c_int, k as c_int);
    let (ldai, ldbi, ldci) = (lda as c_int, ldb as c_int, ldc as c_int);
    let (minus_one, one) = (-1.0f64, 1.0f64);
    /* SAFETY: as `gemm_nt`; `c` is `&mut` where `a` and `b` are shared. */
    unsafe {
        dgemm_(
            &ch(b'N'),
            &ch(b'N'),
            &mi,
            &ni,
            &ki,
            &minus_one,
            a.as_ptr(),
            &ldai,
            b.as_ptr(),
            &ldbi,
            &one,
            c.as_mut_ptr(),
            &ldci,
        );
    }
}

/// `dgemm ("C","N", m, n, k, -1, a, lda, b, ldb, 1, c, ldc)` — solve worker
/// `:494`.
///
/// `a` is `k`-by-`m`, the transpose swapping its extents.
#[allow(clippy::too_many_arguments)]
pub fn gemm_tn(
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    c: &mut [f64],
    ldc: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    let (mi, ni, ki) = (m as c_int, n as c_int, k as c_int);
    let (ldai, ldbi, ldci) = (lda as c_int, ldb as c_int, ldc as c_int);
    let (minus_one, one) = (-1.0f64, 1.0f64);
    /* SAFETY: as `gemm_nn` above. */
    unsafe {
        dgemm_(
            &ch(b'C'),
            &ch(b'N'),
            &mi,
            &ni,
            &ki,
            &minus_one,
            a.as_ptr(),
            &ldai,
            b.as_ptr(),
            &ldbi,
            &one,
            c.as_mut_ptr(),
            &ldci,
        );
    }
}
