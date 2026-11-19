//! R's bundled LAPACK 3.12.1 (`src/modules/lapack/dlapack.f`): the routines
//! behind mgcv's tridiagonal eigensolver (`mgcv_trisymeig`, i.e. `dstedc`)
//! and base R's `eigen(symmetric = TRUE)` (`La_rs`, i.e. `dsyevr`).
//!
//! Each routine is the Fortran statement by statement. Arrays are raw
//! pointers to their first element, indexed from 1 in column-major order
//! (`v!`, `m!`), so that a sub-array argument such as `Z( START, START )`, or
//! a routine handing overlapping pieces of one workspace to its callees,
//! keeps its Fortran meaning.
//!
//! The rounding is that of R's arm64 build (gfortran 14.2 at `-O2`, which
//! contracts across statements): each product gfortran fuses into a
//! multiply-add is one [`fma`] here and every other operation rounds where
//! the Fortran does. BLAS calls go to the vendor library, which is the one
//! R links on macOS (Accelerate).

#![allow(
    clippy::missing_safety_doc,
    clippy::collapsible_else_if,
    clippy::collapsible_if,
    clippy::comparison_chain,
    clippy::explicit_counter_loop,
    clippy::neg_multiply
)]

macro_rules! v {
    ($p:expr, $i:expr) => {
        *$p.offset(($i) as isize - 1)
    };
}

macro_rules! m {
    ($p:expr, $ld:expr, $i:expr, $j:expr) => {
        *$p.offset(($i) as isize - 1 + (($j) as isize - 1) * ($ld) as isize)
    };
}

macro_rules! pv {
    ($p:expr, $i:expr) => {
        $p.offset(($i) as isize - 1)
    };
}

macro_rules! pm {
    ($p:expr, $ld:expr, $i:expr, $j:expr) => {
        $p.offset(($i) as isize - 1 + (($j) as isize - 1) * ($ld) as isize)
    };
}

pub(crate) mod auxiliary;
pub(crate) mod bdsqr;
pub(crate) mod blas;
pub(crate) mod chol;
pub(crate) mod gesdd;
pub(crate) mod larrv;
pub(crate) mod lasd;
pub(crate) mod lasq;
pub(crate) mod lu;
pub(crate) mod ormtr;
pub(crate) mod qr;
pub(crate) mod stedc;
pub(crate) mod stemr;
pub(crate) mod steqr;
pub(crate) mod syevr;
pub(crate) mod sytrd;

/// `a*b + c` as gfortran emits it where it contracts: one rounding on arm64
/// (`fmadd`), two on x86-64, whose baseline has no FMA instruction.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub(crate) fn fma(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

/// LAPACK's `LSAME`: case-insensitive comparison of one character.
#[inline(always)]
pub(crate) fn lsame(a: u8, b: u8) -> bool {
    a.eq_ignore_ascii_case(&b)
}

/// `DLAMCH('E')`: the unit roundoff, `EPSILON(0d0) * 0.5`.
pub(crate) const EPS: f64 = f64::EPSILON * 0.5;
/// `DLAMCH('S')`: the safe minimum, `TINY(0d0)` (its reciprocal does not
/// overflow).
pub(crate) const SAFMIN: f64 = f64::MIN_POSITIVE;
/// `DLAMCH('P')`: `EPS * RADIX`.
pub(crate) const PREC: f64 = f64::EPSILON;
/// `DLAMCH('O')`: `HUGE(0d0)`.
pub(crate) const HUGEVAL: f64 = f64::MAX;

/// Fortran's `SIGN(A, B)`: `|A|` with the sign of `B` (a negative zero `B`
/// counts as negative).
#[inline(always)]
pub(crate) fn sign(a: f64, b: f64) -> f64 {
    a.abs().copysign(b)
}

/// `DISNAN`.
#[inline(always)]
pub(crate) fn disnan(x: f64) -> bool {
    x.is_nan()
}
