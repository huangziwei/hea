//! The platform's BLAS, which `--features blas` asks for and `build.rs` finds.
//!
//! On by default, so a plain build already has it where one is reachable:
//!
//! ```text
//! .venv/bin/maturin develop --release                      # the shipped build
//! .venv/bin/maturin develop --release --no-default-features  # the portable one
//! ```
//!
//! `hea.sparse.build_info()` reports which of the two you got.
//!
//! # The call sequence
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
//! own kernels are one thread per call, so the parallelism has to be built above
//! them. A threaded BLAS supplies it, so this path issues upstream's call
//! sequence unsplit.
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
//! `blas-sweep` is the third of these and the only one that is not shippable:
//! it moves [`MIN_FLOPS`] from a compiled-in constant to a `OnceLock` read of
//! `HEA_BLAS_MIN_FLOPS`, so locating the crossing costs one build instead of one
//! per column. That is what makes the sweep affordable on a CI runner, which is
//! the only x86-64 hardware this project has — see the `bench-min-flops` job in
//! `.github/workflows/python-package.yml` and `.github/bench/min_flops_sweep.py`.
//!
//! # Pinned test values
//!
//! A digest taken against Accelerate is not the one OpenBLAS gives, so in
//! principle a pinned literal stops being one number across platforms. In
//! practice no pin is affected: at [`MIN_FLOPS`] the largest matrix under
//! `tests/` is `n = 1000`, too sparse for any call in it to reach the vendor at
//! all, and the suite passes unchanged on all three arms — Accelerate, OpenBLAS,
//! portable. Should one ever be affected, the shape of the fix is the one hea's
//! other cross-platform pins use: store every value the reference itself
//! produces and assert membership, never branch on the platform.
//!
//! The cost of enabling this by default is installability, which is `build.rs`'s
//! subject: a default feature that can fail to *link* would break `pip install`
//! from the sdist wherever no OpenBLAS is present. Hence the split between the
//! feature (an intent) and `cfg(vendor_blas)` (a backend was found), and
//! `blas-required` for the wheel jobs, where falling back silently would be the
//! defect.

use std::os::raw::{c_char, c_double, c_int};

/// Flops below which a call keeps hea's own kernel instead of the vendor's.
///
/// A sparse factorization is mostly tiny calls: on a 102k-row system 22,685 of
/// the 35,000 descendant updates are eight columns wide and carry 3% of the
/// flops between them. Handed to the vendor they run at ~10 GF/s — that is the
/// dispatch, not the arithmetic — where hea's own kernels do the same work at
/// ~60. The big calls go the other way by a wider margin, ~195 GF/s against 60,
/// because they reach the matrix coprocessor. The two paths are complementary,
/// and this constant is the crossing.
///
/// The eight solve kernels share it. Their regime is a different one — `trsv`
/// and `gemv` are level 2 and move as many bytes as they do flops, while `trsm`
/// and `gemm` widen with the right-hand sides — but the measured crossing lands
/// in the same place, so one number covers all twelve kernels without ever
/// looking at `nrhs`.
///
/// The value is a flat basin rather than a point: on both aarch64 and x86-64,
/// cutoffs from 25 to 1000 kflop are within run-to-run spread of each other.
/// Routing at all is worth 11-22%; routing *everything* (a cutoff of zero) is
/// measurably worse, because then every one of a factorization's thousands of
/// tiny calls pays the vendor's dispatch.
#[cfg(not(feature = "blas-all"))]
pub const MIN_FLOPS: f64 = 1.0e5;

/// Route every call to the vendor. The only build where this path can be
/// checked for bit-exactness.
///
/// The claim this feature tests is that hea on a vendor BLAS issues *upstream's*
/// call sequence — one `dsyrk` and one `dgemm` per descendant, one `dpotrf` and
/// one `dtrsm` per supernode, all full width — rather than the column strips and
/// blocked `potrf` its own kernels need. If that holds, hea linked to Accelerate
/// and CHOLMOD linked to Accelerate do the same arithmetic in the same order and
/// `L->x` must agree to the bit.
///
/// At the shipped cutoff the claim cannot be tested, because calls below the
/// crossing stay on hea's kernels and `L` is then a blend of two libraries'
/// rounding. At zero there is one library and `==` is available; both halves of
/// the pipeline pass it over the whole SPD corpus, both stypes and both
/// orderings, with no tolerance anywhere.
#[cfg(feature = "blas-all")]
pub const MIN_FLOPS: f64 = 0.0;

#[cfg(not(feature = "blas-sweep"))]
#[inline(always)]
pub fn cutoff() -> f64 {
    MIN_FLOPS
}

#[cfg(feature = "blas-sweep")]
pub fn cutoff() -> f64 {
    use std::sync::OnceLock;
    static SWEPT: OnceLock<f64> = OnceLock::new();
    *SWEPT.get_or_init(|| {
        std::env::var("HEA_BLAS_MIN_FLOPS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(MIN_FLOPS)
    })
}

#[inline]
pub fn worth_it(flops: f64) -> bool {
    flops >= cutoff()
}

#[cfg_attr(accelerate, link(name = "Accelerate", kind = "framework"))]
#[cfg_attr(
    all(not(accelerate), target_env = "msvc"),
    link(name = "libscipy_openblas")
)]
#[cfg_attr(
    all(not(accelerate), not(target_env = "msvc")),
    link(name = "scipy_openblas")
)]
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

/// Accelerate gets no such call, and the asymmetry is the two libraries', not a
/// pair of policies.
///
/// OpenBLAS nested under hea's pool is *strictly* worse — 47x the CPU for a 12x
/// worse wall clock — so pinning it is a defect fix. Accelerate nested is
/// something hea deliberately uses: it takes 4.3-5.4 threads inside the pool,
/// and [`super::super_numeric`]'s `PAR_FLOPS` is four orders of magnitude
/// higher on this arm precisely so the wide supernodes go to it whole rather
/// than being cut up by a second scheduler on top of its own.
///
/// Pinning it is possible, so this is a choice rather than a limitation.
/// `BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)` is `thread_local`
/// (`vecLib/thread_api.h`, macOS 15+), so hea can hold one thread on its own
/// workers without touching the caller's numpy, and resolved through
/// `dlsym(RTLD_DEFAULT, …)` it needs no deployment-target bump.
///
/// It is not shipped because the trade is bad where it is visible at all. Set on
/// every pool worker, most systems cannot see it on either axis; the largest
/// ones give up ~10% of the wall clock to save ~22% of the CPU, at roughly nine
/// threads' worth of CPU per wall-second saved. The cost is hea's own schedule
/// being unable to fill the region Accelerate is filling: a schedule that could
/// would take the CPU saving for nothing.
#[cfg(accelerate)]
#[inline]
pub fn init() {}

#[inline]
fn ch(c: u8) -> c_char {
    c as c_char
}

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
