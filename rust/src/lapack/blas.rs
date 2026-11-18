//! The BLAS routines the ported LAPACK calls, issued to the vendor library
//! with LAPACK's own arguments.

use std::os::raw::{c_char, c_double, c_int};

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
    #[cfg_attr(not(accelerate), link_name = "scipy_dcopy_")]
    fn dcopy_(
        n: *const c_int,
        x: *const c_double,
        incx: *const c_int,
        y: *mut c_double,
        incy: *const c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_dscal_")]
    fn dscal_(n: *const c_int, a: *const c_double, x: *mut c_double, incx: *const c_int);
    #[cfg_attr(not(accelerate), link_name = "scipy_dswap_")]
    fn dswap_(
        n: *const c_int,
        x: *mut c_double,
        incx: *const c_int,
        y: *mut c_double,
        incy: *const c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_drot_")]
    fn drot_(
        n: *const c_int,
        x: *mut c_double,
        incx: *const c_int,
        y: *mut c_double,
        incy: *const c_int,
        c: *const c_double,
        s: *const c_double,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_idamax_")]
    fn idamax_(n: *const c_int, x: *const c_double, incx: *const c_int) -> c_int;
    #[cfg_attr(not(accelerate), link_name = "scipy_dnrm2_")]
    fn dnrm2_(n: *const c_int, x: *const c_double, incx: *const c_int) -> c_double;
    #[cfg_attr(not(accelerate), link_name = "scipy_ddot_")]
    fn ddot_(
        n: *const c_int,
        x: *const c_double,
        incx: *const c_int,
        y: *const c_double,
        incy: *const c_int,
    ) -> c_double;
    #[cfg_attr(not(accelerate), link_name = "scipy_daxpy_")]
    fn daxpy_(
        n: *const c_int,
        a: *const c_double,
        x: *const c_double,
        incx: *const c_int,
        y: *mut c_double,
        incy: *const c_int,
    );
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
    #[cfg_attr(not(accelerate), link_name = "scipy_dger_")]
    fn dger_(
        m: *const c_int,
        n: *const c_int,
        alpha: *const c_double,
        x: *const c_double,
        incx: *const c_int,
        y: *const c_double,
        incy: *const c_int,
        a: *mut c_double,
        lda: *const c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_dsymv_")]
    fn dsymv_(
        uplo: *const c_char,
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
    #[cfg_attr(not(accelerate), link_name = "scipy_dsyr2_")]
    fn dsyr2_(
        uplo: *const c_char,
        n: *const c_int,
        alpha: *const c_double,
        x: *const c_double,
        incx: *const c_int,
        y: *const c_double,
        incy: *const c_int,
        a: *mut c_double,
        lda: *const c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_dsyr2k_")]
    fn dsyr2k_(
        uplo: *const c_char,
        trans: *const c_char,
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
    #[cfg_attr(not(accelerate), link_name = "scipy_dtrmv_")]
    fn dtrmv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const c_int,
        a: *const c_double,
        lda: *const c_int,
        x: *mut c_double,
        incx: *const c_int,
    );
    #[cfg_attr(not(accelerate), link_name = "scipy_dtrmm_")]
    fn dtrmm_(
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
}

#[inline(always)]
fn ch(c: u8) -> c_char {
    c as c_char
}

pub(crate) unsafe fn dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32) {
    dcopy_(&n, x, &incx, y, &incy);
}

pub(crate) unsafe fn dscal(n: i32, a: f64, x: *mut f64, incx: i32) {
    dscal_(&n, &a, x, &incx);
}

pub(crate) unsafe fn dswap(n: i32, x: *mut f64, incx: i32, y: *mut f64, incy: i32) {
    dswap_(&n, x, &incx, y, &incy);
}

pub(crate) unsafe fn drot(n: i32, x: *mut f64, incx: i32, y: *mut f64, incy: i32, c: f64, s: f64) {
    drot_(&n, x, &incx, y, &incy, &c, &s);
}

pub(crate) unsafe fn idamax(n: i32, x: *const f64, incx: i32) -> i32 {
    idamax_(&n, x, &incx)
}

pub(crate) unsafe fn dnrm2(n: i32, x: *const f64, incx: i32) -> f64 {
    dnrm2_(&n, x, &incx)
}

pub(crate) unsafe fn ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64 {
    ddot_(&n, x, &incx, y, &incy)
}

pub(crate) unsafe fn daxpy(n: i32, a: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32) {
    daxpy_(&n, &a, x, &incx, y, &incy);
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    dgemm_(
        &ch(transa),
        &ch(transb),
        &m,
        &n,
        &k,
        &alpha,
        a,
        &lda,
        b,
        &ldb,
        &beta,
        c,
        &ldc,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dgemv(
    trans: u8,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    dgemv_(
        &ch(trans),
        &m,
        &n,
        &alpha,
        a,
        &lda,
        x,
        &incx,
        &beta,
        y,
        &incy,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dger(
    m: i32,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    a: *mut f64,
    lda: i32,
) {
    dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dsymv(
    uplo: u8,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    x: *const f64,
    incx: i32,
    beta: f64,
    y: *mut f64,
    incy: i32,
) {
    dsymv_(&ch(uplo), &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dsyr2(
    uplo: u8,
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    a: *mut f64,
    lda: i32,
) {
    dsyr2_(&ch(uplo), &n, &alpha, x, &incx, y, &incy, a, &lda);
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dsyrk(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    dsyrk_(
        &ch(uplo),
        &ch(trans),
        &n,
        &k,
        &alpha,
        a,
        &lda,
        &beta,
        c,
        &ldc,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dsyr2k(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    dsyr2k_(
        &ch(uplo),
        &ch(trans),
        &n,
        &k,
        &alpha,
        a,
        &lda,
        b,
        &ldb,
        &beta,
        c,
        &ldc,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dtrmv(
    uplo: u8,
    trans: u8,
    diag: u8,
    n: i32,
    a: *const f64,
    lda: i32,
    x: *mut f64,
    incx: i32,
) {
    dtrmv_(&ch(uplo), &ch(trans), &ch(diag), &n, a, &lda, x, &incx);
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dtrmm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *mut f64,
    ldb: i32,
) {
    dtrmm_(
        &ch(side),
        &ch(uplo),
        &ch(transa),
        &ch(diag),
        &m,
        &n,
        &alpha,
        a,
        &lda,
        b,
        &ldb,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dtrsm(
    side: u8,
    uplo: u8,
    transa: u8,
    diag: u8,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *mut f64,
    ldb: i32,
) {
    dtrsm_(
        &ch(side),
        &ch(uplo),
        &ch(transa),
        &ch(diag),
        &m,
        &n,
        &alpha,
        a,
        &lda,
        b,
        &ldb,
    );
}
