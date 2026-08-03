//! Fisher's exact test for r×c contingency tables — the Mehta-Patel FEXACT
//! network algorithm (ACM TOMS 643).
//!
//! A line-by-line mirror of the pure-Python `hea/R/_fexact.py`, which is itself
//! a faithful port of R's `src/library/stats/src/fexact.c`. The Python module
//! is the spec and the test oracle; `tests/test_rs_parity.py` pins this Rust
//! kernel `== python` (transitively `== R`, since Python is pinned bit-exact to
//! R). The algorithm is deterministic double arithmetic — the log-factorial
//! table, the shortest/longest path bounds (f3xact/f4xact), and the `pre`
//! accumulation reproduce R's operation order, so the p-value is bit-exact.
//!
//! The hash-table sizes `ldkey`/`ldstp` are derived from `workspace`/`mult`
//! exactly as R's `iwork()` does, because the `pre`-accumulation order (hence
//! the last ulps) depends on them.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::nmath::gamma::pgamma_scalar;

const INT_MAX: i64 = 2147483647;
const TOL: f64 = 3.45254e-7;
const AMISS: f64 = -12345.0;
const LOG_2PI: f64 = 1.83787706640934548356065947281;

/// 1-based (index 0 unused) vector with `i64` indexing so the port reads like
/// its Python spec.
struct IV(Vec<i64>);
struct FV(Vec<f64>);

impl std::ops::Index<i64> for IV {
    type Output = i64;
    #[inline]
    fn index(&self, i: i64) -> &i64 {
        &self.0[i as usize]
    }
}
impl std::ops::IndexMut<i64> for IV {
    #[inline]
    fn index_mut(&mut self, i: i64) -> &mut i64 {
        &mut self.0[i as usize]
    }
}
impl std::ops::Index<i64> for FV {
    type Output = f64;
    #[inline]
    fn index(&self, i: i64) -> &f64 {
        &self.0[i as usize]
    }
}
impl std::ops::IndexMut<i64> for FV {
    #[inline]
    fn index_mut(&mut self, i: i64) -> &mut f64 {
        &mut self.0[i as usize]
    }
}

#[inline]
fn imax(a: i64, b: i64) -> i64 {
    if a > b {
        a
    } else {
        b
    }
}
#[inline]
fn imin(a: i64, b: i64) -> i64 {
    if a < b {
        a
    } else {
        b
    }
}

/// f11act — revise row totals: copy `src` column to `dst` dropping the entry at
/// 1-based position `i1` (`arr` flat, columns pre-offset).
fn f11act(arr: &mut [i64], src: i64, dst: i64, i1: i64, i2: i64) {
    let mut m = 0;
    while m < i1 - 1 {
        arr[(dst + m) as usize] = arr[(src + m) as usize];
        m += 1;
    }
    let mut m = i1;
    while m <= i2 {
        arr[(dst + m - 1) as usize] = arr[(src + m) as usize];
        m += 1;
    }
}

/// f8xact — reduce a vector with a zero element: copy `src` to `dst` inserting
/// `is_` at its sorted position (both columns pre-offset, 1-based).
fn f8act(arr: &mut [i64], src: i64, dst: i64, is_: i64, i1: i64, izero: i64) {
    let mut i = 1;
    while i < i1 {
        arr[(dst + i - 1) as usize] = arr[(src + i - 1) as usize];
        i += 1;
    }
    while i <= izero - 1 {
        if is_ >= arr[(src + i) as usize] {
            break;
        }
        arr[(dst + i - 1) as usize] = arr[(src + i) as usize];
        i += 1;
    }
    arr[(dst + i - 1) as usize] = is_;
    loop {
        i += 1;
        if i > izero {
            return;
        }
        arr[(dst + i - 1) as usize] = arr[(src + i - 1) as usize];
    }
}

/// f9xact — log of a multinomial coefficient `log(ntot!) - sum log(ir!)`.
fn f9xact(n: i64, ntot: i64, ir: &[i64], fact: &FV) -> f64 {
    let mut d = fact[ntot];
    for k in 0..n {
        d -= fact[ir[k as usize]];
    }
    d
}

/// f10act — shortest path length for special tables (`irow`/`icol` 0-based).
/// Returns `(xmin, val)`.
fn f10act(
    nrow: i64,
    irow: &[i64],
    ncol: i64,
    icol: &[i64],
    mut val: f64,
    fact: &FV,
) -> (bool, f64) {
    let nr = nrow as usize;
    let nc = ncol as usize;
    let mut nd = vec![0i64; nr];
    let mut ne = vec![0i64; nc];
    let mut m = vec![0i64; nc];

    let mut is_ = icol[0] / nrow;
    let mut ix = icol[0] - nrow * is_;
    ne[0] = is_;
    m[0] = ix;
    if ix != 0 {
        nd[(ix - 1) as usize] += 1;
    }
    for i in 1..nc {
        ix = icol[i] / nrow;
        ne[i] = ix;
        is_ += ix;
        ix = icol[i] - nrow * ix;
        m[i] = ix;
        if ix != 0 {
            nd[(ix - 1) as usize] += 1;
        }
    }
    let mut i = nrow - 3;
    while i >= 0 {
        nd[i as usize] += nd[(i + 1) as usize];
        i -= 1;
    }
    ix = 0;
    let mut i = nrow;
    while i >= 2 {
        ix += is_ + nd[(nrow - i) as usize] - irow[(i - 1) as usize];
        if ix < 0 {
            return (false, val);
        }
        i -= 1;
    }
    for i in 0..nc {
        ix = ne[i];
        is_ = m[i];
        val += is_ as f64 * fact[ix + 1] + (nrow - is_) as f64 * fact[ix];
    }
    (true, val)
}

struct Fexact {
    nrow: i64,
    ncol: i64,
    m: Vec<i64>, // row-major flat, TBL(i,j) = m[(i-1)*ncol + (j-1)]
    expect: f64,
    percnt: f64,
    emin: f64,
    workspace: i64,
    mult: i64,
    // sized in run()
    ldkey: i64,
    ldstp: i64,
    n2_stack: i64,
    // work arrays (1-based; index 0 sentinel)
    iro: IV,
    ico: IV,
    kyy: IV,
    idif: IV,
    irn: IV,
    key: IV,
    key2: IV,
    ipoin: IV,
    lp: FV,
    sp: FV,
    tm: FV,
    stp: FV,
    ifrq: IV,
    fact: FV,
    // C statics
    f3_nst: i64,
    f3_nitc: i64,
    f5_itp: i64,
}

impl Fexact {
    fn tbl(&self, i: i64, j: i64) -> i64 {
        self.m[((i - 1) * self.ncol + (j - 1)) as usize]
    }

    fn run(&mut self) -> Result<f64, String> {
        let nrow = self.nrow;
        let ncol = self.ncol;
        let mut ntot = 0i64;
        for i in 0..nrow {
            for j in 0..ncol {
                let v = self.m[(i * ncol + j) as usize];
                if v < 0 {
                    return Err("All elements of TABLE must be nonnegative.".into());
                }
                ntot += v;
            }
        }
        if ntot == 0 {
            return Ok(AMISS);
        }
        let nco = imax(nrow, ncol);
        let nro = imin(nrow, ncol);
        let k = nrow + ncol + 1;
        let kk = k * nco;
        let iwkmax = 2 * (self.workspace / 2);
        let n2_stack = imax(200, iwkmax / 1000);

        // Reproduce iwork()'s accounting up to the hash tables so ldkey/ldstp
        // match R exactly (the pre-accumulation order depends on them).
        let mut iwkpt = 0i64;
        iwkpt += (ntot + 1) << 1; // i1  fact (double)
        iwkpt += nco; // i2  ico
        iwkpt += nco; // i3  iro
        iwkpt += nco; // i3a kyy
        iwkpt += nro; // i3b idif
        iwkpt += nro; // i3c irn
        iwkpt += imax(k * 5 + (kk << 1), nco * 7 + 4 * n2_stack); // iiwk
        iwkpt += imax(nco + 1 + 2 * n2_stack, k) << 1; // irwk (double)

        let numb = 18 + 10 * self.mult;
        let ldkey = (iwkmax - iwkpt) / numb - 1;
        if ldkey < 1 {
            return Err("workspace too small for this table; increase 'workspace'".into());
        }
        if self.mult * ldkey > INT_MAX {
            return Err("integer overflow would happen in 'mult * ldkey'".into());
        }
        let ldstp = self.mult * ldkey;
        self.ldkey = ldkey;
        self.ldstp = ldstp;
        self.n2_stack = n2_stack;

        let z = (nco + 2) as usize;
        self.iro = IV(vec![0; z]);
        self.ico = IV(vec![0; z]);
        self.kyy = IV(vec![0; z]);
        self.idif = IV(vec![0; z]);
        self.irn = IV(vec![0; z]);
        self.key = IV(vec![-9999; (2 * ldkey + 1) as usize]);
        self.key2 = IV(vec![-9999; (2 * ldkey + 1) as usize]);
        self.ipoin = IV(vec![0; (2 * ldkey + 1) as usize]);
        self.lp = FV(vec![0.0; (2 * ldkey + 1) as usize]);
        self.sp = FV(vec![0.0; (2 * ldkey + 1) as usize]);
        self.tm = FV(vec![0.0; (2 * ldkey + 1) as usize]);
        self.stp = FV(vec![0.0; (2 * ldstp + 1) as usize]);
        self.ifrq = IV(vec![0; (6 * ldstp + 1) as usize]);

        self.f2xact()
    }

    fn f2xact(&mut self) -> Result<f64, String> {
        let nrow = self.nrow;
        let ncol = self.ncol;
        let tol = TOL;
        let maybe_chisq = self.expect > 0.0;
        let expect = self.expect;
        let percnt = self.percnt;
        let emin = self.emin;
        let ldkey = self.ldkey;
        let ldstp = self.ldstp;

        let nr_gt_nc = nrow > ncol;
        let nco = if nr_gt_nc { nrow } else { ncol };

        // Row marginals + total
        let mut ntot = 0i64;
        for i in 1..=nrow {
            self.iro[i] = 0;
            for j in 1..=ncol {
                self.iro[i] += self.tbl(i, j);
            }
            ntot += self.iro[i];
        }
        // Column marginals
        for i in 1..=ncol {
            self.ico[i] = 0;
            for j in 1..=nrow {
                self.ico[i] += self.tbl(j, i);
            }
        }
        self.iro.0[1..=nrow as usize].sort_unstable();
        self.ico.0[1..=ncol as usize].sort_unstable();

        let nro;
        if nr_gt_nc {
            nro = ncol;
            for i in 1..=nco {
                let ii = self.iro[i];
                if i <= nro {
                    self.iro[i] = self.ico[i];
                }
                self.ico[i] = ii;
            }
        } else {
            nro = nrow;
        }

        // Hash-table multipliers
        self.kyy[1] = 1;
        for i in 1..nro {
            if self.iro[i] + 1 <= INT_MAX / self.kyy[i] {
                self.kyy[i + 1] = self.kyy[i] * (self.iro[i] + 1);
            } else {
                return Err(
                    "the hash key would exceed the largest representable int; consider \
                     using 'simulate_p_value=True'"
                        .into(),
                );
            }
        }
        if self.iro[nro] + 1 > INT_MAX / self.kyy[nro] {
            return Err(
                "the hash key would exceed the largest representable int; consider \
                 using 'simulate_p_value=True'"
                    .into(),
            );
        }

        // Log factorials (R's exact recurrence, not lgamma)
        self.fact = FV(vec![0.0; (ntot + 1) as usize]);
        if ntot >= 2 {
            self.fact[2] = 2.0f64.ln();
        }
        let mut i = 3;
        while i <= ntot {
            self.fact[i] = self.fact[i - 1] + (i as f64).ln();
            let j = i + 1;
            if j <= ntot {
                self.fact[j] =
                    self.fact[i] + self.fact[2] + self.fact[j / 2] - self.fact[j / 2 - 1];
            }
            i += 2;
        }

        // Observed path length
        let mut obs = tol;
        ntot = 0;
        for j in 1..=nco {
            let mut dd = 0.0;
            if nr_gt_nc {
                for i in 1..=nro {
                    dd += self.fact[self.tbl(j, i)];
                    ntot += self.tbl(j, i);
                }
            } else {
                for i in 1..=nro {
                    dd += self.fact[self.tbl(i, j)];
                    ntot += self.tbl(i, j);
                }
            }
            obs += self.fact[self.ico[j]] - dd;
        }

        let dro = f9xact(nro, ntot, &self.iro.0[1..], &self.fact);
        let mut pre = 0.0;
        let mut itop = 0i64;

        // Buffer / stage pointers
        let mut k = nco;
        let mut last = ldkey + 1;
        let mut jkey = ldkey + 1;
        let mut jstp = ldstp + 1;
        let mut jstp2 = ldstp * 3 + 1;
        let jstp3 = (ldstp << 2) + 1;
        let jstp4 = ldstp * 5 + 1;
        let mut ikkey = 0i64;
        let mut ikstp = 0i64;
        let mut ikstp2 = ldstp << 1;
        let mut ipo = 1i64;
        self.ipoin[1] = 1;
        self.stp[1] = 0.0;
        self.ifrq[1] = 1;
        self.ifrq[ikstp2 + 1] = -1;

        // Per-node state
        let mut k1 = 0i64;
        let mut nro2 = 0i64;
        let mut nrb = 0i64;
        let mut ddf = 0.0;
        let mut drn = 0.0;
        let mut obs2 = 0.0;
        let mut obs3 = 0.0;
        let mut tmp = 0.0;
        let mut kval = 0i64;
        let mut itp = 0i64;
        let mut chisq = false;
        let mut kmax = 0i64;
        let mut kd = 0i64;
        let mut ks = 0i64;
        let mut n = 0i64;
        let mut ifreq = 0i64;
        let mut pastp = 0.0;
        let mut ipn = 0i64;
        let mut psh = true;

        let mut state = "Outer_Loop";
        loop {
            match state {
                "Outer_Loop" => {
                    let kb = nco - k + 1;
                    ks = 0;
                    n = self.ico[kb];
                    kd = nro + 1;
                    kmax = nro;
                    for i in 1..=nro {
                        self.idif[i] = 0;
                    }
                    loop {
                        kd -= 1;
                        ntot = imin(n, self.iro[kd]);
                        self.idif[kd] = ntot;
                        if self.idif[kmax] == 0 {
                            kmax -= 1;
                        }
                        n -= ntot;
                        if !(n > 0 && kd != 1) {
                            break;
                        }
                    }
                    if n != 0 {
                        state = "L310";
                        continue;
                    }
                    k1 = k - 1;
                    n = self.ico[kb];
                    ntot = 0;
                    for i in (kb + 1)..=nco {
                        ntot += self.ico[i];
                    }
                    state = "L150";
                    continue;
                }
                "L150" => {
                    for i in 1..=nro {
                        self.irn[i] = self.iro[i] - self.idif[i];
                    }
                    if k1 > 1 {
                        self.irn.0[1..=nro as usize].sort_unstable();
                        nrb = 1;
                        for i in 1..=nro {
                            if self.irn[i] != 0 {
                                nrb = i;
                                break;
                            }
                            if i == nro {
                                nrb = nro + 1;
                            }
                        }
                    } else {
                        nrb = 1;
                    }
                    nro2 = nro - nrb + 1;

                    ddf = f9xact(nro, n, &self.idif.0[1..], &self.fact);
                    drn = f9xact(nro2, ntot, &self.irn.0[nrb as usize..], &self.fact) - dro + ddf;

                    if k1 > 1 {
                        kval = self.irn[1];
                        for i in 2..=nro {
                            kval += self.irn[i] * self.kyy[i];
                        }
                        let ii_hash = kval % (ldkey << 1) + 1;
                        let mut found = false;
                        let mut t = ii_hash;
                        while t <= (ldkey << 1) {
                            let ii = self.key2[t];
                            if ii == kval {
                                itp = t;
                                found = true;
                                break;
                            } else if ii < 0 {
                                self.key2[t] = kval;
                                self.lp[t] = 1.0;
                                self.sp[t] = 1.0;
                                itp = t;
                                found = true;
                                break;
                            }
                            t += 1;
                        }
                        if !found {
                            let mut t = 1;
                            while t < ii_hash {
                                let ii = self.key2[t];
                                if ii == kval {
                                    itp = t;
                                    found = true;
                                    break;
                                } else if ii < 0 {
                                    self.key2[t] = kval;
                                    self.lp[t] = 1.0;
                                    itp = t;
                                    found = true;
                                    break;
                                }
                                t += 1;
                            }
                        }
                        if !found {
                            return Err(format!(
                                "FEXACT error 6: LDKEY={} too small; increase 'workspace' \
                                 (or use simulate_p_value=True)",
                                ldkey
                            ));
                        }
                    }
                    state = "L240";
                    continue;
                }
                "L240" => {
                    let kb = nco - k + 1;
                    psh = true;
                    ipn = self.ipoin[ipo + ikkey];
                    pastp = self.stp[ipn + ikstp];
                    ifreq = self.ifrq[ipn + ikstp];
                    if k1 > 1 {
                        obs2 =
                            obs - self.fact[self.ico[kb + 1]] - self.fact[self.ico[kb + 2]] - ddf;
                        for i in 3..=k1 {
                            obs2 -= self.fact[self.ico[kb + i]];
                        }
                        if self.lp[itp] > 0.0 {
                            let dspt = obs - obs2 - ddf;
                            let irow_s: Vec<i64> =
                                self.irn.0[nrb as usize..(nrb + nro2) as usize].to_vec();
                            let icol_s: Vec<i64> =
                                self.ico.0[(kb + 1) as usize..(kb + 1 + k1) as usize].to_vec();
                            let lpv =
                                self.f3xact(nro2, &irow_s, k1, &icol_s, ntot, self.n2_stack)?;
                            self.lp[itp] = lpv;
                            if self.lp[itp] > 0.0 {
                                self.lp[itp] = 0.0;
                            }
                            let spv = self.f4xact(nro2, &irow_s, k1, &icol_s, dspt);
                            self.sp[itp] = spv;
                            if self.sp[itp] > 0.0 {
                                self.sp[itp] = 0.0;
                            }
                            if maybe_chisq
                                && (self.irn[nrb] * self.ico[kb + 1]) as f64 > ntot as f64 * emin
                            {
                                let mut ncell = 0i64;
                                for i in 0..nro2 {
                                    for j in 1..=k1 {
                                        if (self.irn[nrb + i] * self.ico[kb + j]) as f64
                                            >= ntot as f64 * expect
                                        {
                                            ncell += 1;
                                        }
                                    }
                                }
                                if (ncell * 100) as f64 >= (k1 * nro2) as f64 * percnt {
                                    tmp = 0.0;
                                    for i in 0..nro2 {
                                        tmp += self.fact[self.irn[nrb + i]]
                                            - self.fact[self.irn[nrb + i] - 1];
                                    }
                                    tmp *= (k1 - 1) as f64;
                                    for j in 1..=k1 {
                                        tmp += (nro2 - 1) as f64
                                            * (self.fact[self.ico[kb + j]]
                                                - self.fact[self.ico[kb + j] - 1]);
                                    }
                                    let df = ((nro2 - 1) * (k1 - 1)) as f64;
                                    tmp += df * LOG_2PI;
                                    tmp -= (nro2 * k1 - 1) as f64
                                        * (self.fact[ntot] - self.fact[ntot - 1]);
                                    self.tm[itp] = (obs - dro) * -2.0 - tmp;
                                } else {
                                    self.tm[itp] = -9876.0;
                                }
                            } else {
                                self.tm[itp] = -9876.0;
                            }
                        }
                        obs3 = obs2 - self.lp[itp];
                        obs2 -= self.sp[itp];
                        if self.tm[itp] == -9876.0 {
                            chisq = false;
                        } else {
                            chisq = true;
                            tmp = self.tm[itp];
                        }
                    } else {
                        obs2 = obs - drn - dro;
                        obs3 = obs2;
                    }
                    state = "L300";
                    continue;
                }
                "L300" => {
                    if pastp <= obs3 {
                        pre += ifreq as f64 * (pastp + drn).exp();
                    } else if pastp < obs2 {
                        if chisq {
                            let df = ((nro2 - 1) * (k1 - 1)) as f64;
                            let pv = pgamma_scalar(
                                (0.0f64).max(tmp + (pastp + drn) * 2.0) / 2.0,
                                df / 2.0,
                                1.0,
                                false,
                                true,
                            );
                            pre += ifreq as f64 * (pastp + drn + pv).exp();
                        } else {
                            itop = self.f5xact(
                                pastp + ddf,
                                kval,
                                ifreq,
                                itop,
                                jkey,
                                jstp,
                                jstp2,
                                jstp3,
                                jstp4,
                                psh,
                            )?;
                            psh = false;
                        }
                    }
                    ipn = self.ifrq[ipn + ikstp2];
                    if ipn > 0 {
                        pastp = self.stp[ipn + ikstp];
                        ifreq = self.ifrq[ipn + ikstp];
                        state = "L300";
                        continue;
                    }
                    let (ok_f7, nk, nks) = self.f7xact(kmax, kd, ks);
                    kd = nk;
                    ks = nks;
                    if ok_f7 {
                        state = "L150";
                        continue;
                    }
                    state = "L310";
                    continue;
                }
                "L310" => {
                    loop {
                        let (done, nlast, nipo) = self.f6xact(nro, ikkey, ldkey, last);
                        last = nlast;
                        ipo = nipo;
                        if !done {
                            state = "Outer_Loop";
                            break;
                        }
                        k -= 1;
                        itop = 0;
                        ikkey = jkey - 1;
                        ikstp = jstp - 1;
                        ikstp2 = jstp2 - 1;
                        jkey = ldkey - jkey + 2;
                        jstp = ldstp - jstp + 2;
                        jstp2 = (ldstp << 1) + jstp;
                        for i in 1..=(ldkey << 1) {
                            self.key2[i] = -9999;
                        }
                        if k < 2 {
                            return Ok(pre);
                        }
                    }
                    continue;
                }
                _ => unreachable!(),
            }
        }
    }

    fn f3xact(
        &mut self,
        nrow: i64,
        irow_s: &[i64],
        ncol: i64,
        icol_s: &[i64],
        ntot: i64,
        ldst: i64,
    ) -> Result<f64, String> {
        // 1-based views (index 0 dummy)
        let mut irow = IV(Vec::with_capacity(irow_s.len() + 1));
        irow.0.push(0);
        irow.0.extend_from_slice(irow_s);
        let mut icol = IV(Vec::with_capacity(icol_s.len() + 1));
        icol.0.push(0);
        icol.0.extend_from_slice(icol_s);

        if nrow <= 1 {
            let mut lp = 0.0;
            if nrow > 0 {
                for i in 1..=ncol {
                    lp -= self.fact[icol[i]];
                }
            }
            return Ok(lp);
        }
        if ncol <= 1 {
            let mut lp = 0.0;
            if ncol > 0 {
                for i in 1..=nrow {
                    lp -= self.fact[irow[i]];
                }
            }
            return Ok(lp);
        }
        if nrow * ncol == 4 {
            let n11 = (irow[1] + 1) * (icol[1] + 1) / (ntot + 2);
            let n12 = irow[1] - n11;
            return Ok(-(self.fact[n11]
                + self.fact[n12]
                + self.fact[icol[1] - n11]
                + self.fact[icol[2] - n12]));
        }

        let mut val = 0.0;
        let mut xmin;
        if irow[nrow] <= irow[1] + ncol {
            let (x, v) = f10act(nrow, &irow.0[1..], ncol, &icol.0[1..], val, &self.fact);
            xmin = x;
            val = v;
        } else {
            xmin = false;
        }
        if !xmin && icol[ncol] <= icol[1] + nrow {
            let (x, v) = f10act(ncol, &icol.0[1..], nrow, &irow.0[1..], val, &self.fact);
            xmin = x;
            val = v;
        }
        if xmin {
            return Ok(-val);
        }

        let mx = imax(nrow, ncol);
        let sz = (mx + 2) as usize;
        let mut ico = IV(vec![0; sz]);
        let mut iro = IV(vec![0; sz]);
        let mut it = IV(vec![0; sz]);
        let mut lb = IV(vec![0; sz]);
        let mut nr = IV(vec![0; sz]);
        let mut nt = IV(vec![0; sz]);
        let mut nu = IV(vec![0; sz]);
        let mut alen = FV(vec![0.0; (ncol + 2) as usize]);
        let mut itc = IV(vec![0; (2 * ldst + 1) as usize]);
        let mut ist = IV(vec![-1; (2 * ldst + 1) as usize]);
        let mut stv = FV(vec![0.0; (2 * ldst + 1) as usize]);

        let nn0 = ntot;
        let mut nn = nn0;
        let nro;
        let nco;
        if nrow >= ncol {
            nro = nrow;
            nco = ncol;
            ico[1] = icol[1];
            nt[1] = nn - ico[1];
            for i in 2..=ncol {
                ico[i] = icol[i];
                nt[i] = nt[i - 1] - ico[i];
            }
            for i in 1..=nrow {
                iro[i] = irow[i];
            }
        } else {
            nro = ncol;
            nco = nrow;
            ico[1] = irow[1];
            nt[1] = nn - ico[1];
            for i in 2..=nrow {
                ico[i] = irow[i];
                nt[i] = nt[i - 1] - ico[i];
            }
            for i in 1..=ncol {
                iro[i] = icol[i];
            }
        }

        let nc1s = nco - 1;
        let kyy = ico[nco] + 1;
        let mut irl = 1i64;
        let mut ks = 0i64;
        let mut k = ldst;
        let mut vmn = 1e100;
        let mut nro = nro;
        let mut lev = 0i64;
        let mut nr1 = 0i64;

        let mut state = "LnewNode";
        loop {
            match state {
                "LnewNode" => {
                    lev = 1;
                    nr1 = nro - 1;
                    let nrt = iro[irl];
                    let nct = ico[1];
                    lb[1] = (((nrt + 1) as f64 * (nct + 1) as f64) / (nn + nr1 * nc1s + 1) as f64
                        - TOL) as i64
                        - 1;
                    nu[1] = ((nrt + nc1s) as f64 * (nct + nr1) as f64 / (nn + nr1 + nc1s) as f64)
                        as i64
                        - lb[1]
                        + 1;
                    nr[1] = nrt - lb[1];
                    state = "LoopNode";
                    continue;
                }
                "LoopNode" => {
                    nu[lev] -= 1;
                    if nu[lev] == 0 {
                        if lev == 1 {
                            state = "L200";
                            continue;
                        }
                        lev -= 1;
                        state = "LoopNode";
                        continue;
                    }
                    lb[lev] += 1;
                    nr[lev] -= 1;
                    loop {
                        alen[lev] = alen[lev - 1] + self.fact[lb[lev]];
                        if lev >= nc1s {
                            break;
                        }
                        let nn1 = nt[lev];
                        let nrt = nr[lev];
                        lev += 1;
                        let nc1 = nco - lev;
                        let nct = ico[lev];
                        lb[lev] = ((nrt + 1) as f64 * (nct + 1) as f64
                            / (nn1 + nr1 * nc1 + 1) as f64
                            - TOL) as i64;
                        nu[lev] = ((nrt + nc1) as f64 * (nct + nr1) as f64
                            / (nn1 + nr1 + nc1) as f64
                            - lb[lev] as f64
                            + 1.0) as i64;
                        nr[lev] = nrt - lb[lev];
                    }
                    alen[nco] = alen[lev] + self.fact[nr[lev]];
                    lb[nco] = nr[lev];
                    let mut v = val + alen[nco];
                    if nro == 2 {
                        v += self.fact[ico[1] - lb[1]] + self.fact[ico[2] - lb[2]];
                        for i in 3..=nco {
                            v += self.fact[ico[i] - lb[i]];
                        }
                        if vmn > v {
                            vmn = v;
                        }
                        state = "LoopNode";
                        continue;
                    } else if nro == 3 && nco == 2 {
                        let nn1 = nn - iro[irl] + 2;
                        let ic1 = ico[1] - lb[1];
                        let ic2 = ico[2] - lb[2];
                        let n11 = (iro[irl + 1] + 1) * (ic1 + 1) / nn1;
                        let n12 = iro[irl + 1] - n11;
                        v += self.fact[n11]
                            + self.fact[n12]
                            + self.fact[ic1 - n11]
                            + self.fact[ic2 - n12];
                        if vmn > v {
                            vmn = v;
                        }
                        state = "LoopNode";
                        continue;
                    } else {
                        for i in 1..=nco {
                            it[i] = imax(ico[i] - lb[i], 0);
                        }
                        it.0[1..=nco as usize].sort_unstable();
                        let dky = kyy as f64;
                        let mut dkey = it[1] as f64 * dky + it[2] as f64;
                        for i in 3..=nco {
                            dkey = it[i] as f64 + dkey * dky;
                        }
                        if dkey > INT_MAX as f64 {
                            return Err("FEXACT[f3xact] hash key exceeds INT_MAX; use \
                                 simulate_p_value=True"
                                .into());
                        }
                        let key = dkey as i64;
                        let ipn = key % ldst + 1;
                        let mut pushed = false;
                        let mut t = ipn;
                        while t <= ldst {
                            let ii = ks + t;
                            if ist[ii] < 0 {
                                ist[ii] = key;
                                stv[ii] = v;
                                self.f3_nst += 1;
                                itc[self.f3_nst + ks] = t;
                                pushed = true;
                                break;
                            } else if ist[ii] == key {
                                if v < stv[ii] {
                                    stv[ii] = v;
                                }
                                pushed = true;
                                break;
                            }
                            t += 1;
                        }
                        if !pushed {
                            let mut t = 1;
                            while t < ipn {
                                let ii = ks + t;
                                if ist[ii] < 0 {
                                    ist[ii] = key;
                                    stv[ii] = v;
                                    self.f3_nst += 1;
                                    itc[self.f3_nst + ks] = t;
                                    pushed = true;
                                    break;
                                } else if ist[ii] == key {
                                    if v < stv[ii] {
                                        stv[ii] = v;
                                    }
                                    pushed = true;
                                    break;
                                }
                                t += 1;
                            }
                        }
                        if !pushed {
                            return Err(format!(
                                "FEXACT error 30: stack length exceeded in f3xact (ldst={}); \
                                 increase 'workspace' (or use simulate_p_value=True)",
                                ldst
                            ));
                        }
                        state = "LoopNode";
                        continue;
                    }
                }
                "L200" => {
                    if self.f3_nitc > 0 {
                        let itp = itc[self.f3_nitc + k] + k;
                        self.f3_nitc -= 1;
                        val = stv[itp];
                        let mut key = ist[itp];
                        ist[itp] = -1;
                        let mut i = nco;
                        while i >= 2 {
                            ico[i] = key % kyy;
                            key /= kyy;
                            i -= 1;
                        }
                        ico[1] = key;
                        nt[1] = nn - ico[1];
                        for i in 2..=nco {
                            nt[i] = nt[i - 1] - ico[i];
                        }
                        let mut xmin;
                        if iro[nro] <= iro[irl] + nco {
                            let (x, v) = f10act(
                                nro,
                                &iro.0[irl as usize..],
                                nco,
                                &ico.0[1..],
                                val,
                                &self.fact,
                            );
                            xmin = x;
                            val = v;
                        } else {
                            xmin = false;
                        }
                        if !xmin && ico[nco] <= ico[1] + nro {
                            let (x, v) = f10act(
                                nco,
                                &ico.0[1..],
                                nro,
                                &iro.0[irl as usize..],
                                val,
                                &self.fact,
                            );
                            xmin = x;
                            val = v;
                        }
                        if xmin {
                            if vmn > val {
                                vmn = val;
                            }
                            state = "L200";
                            continue;
                        } else {
                            state = "LnewNode";
                            continue;
                        }
                    } else if nro > 2 && self.f3_nst > 0 {
                        self.f3_nitc = self.f3_nst;
                        self.f3_nst = 0;
                        k = ks;
                        ks = ldst - ks;
                        nn -= iro[irl];
                        irl += 1;
                        nro -= 1;
                        state = "L200";
                        continue;
                    }
                    return Ok(-vmn);
                }
                _ => unreachable!(),
            }
        }
    }

    fn f4xact(&mut self, nrow: i64, irow_s: &[i64], ncol: i64, icol_s: &[i64], dspt: f64) -> f64 {
        let tol = TOL;
        let irow = irow_s;
        let icol = icol_s;

        if nrow == 1 {
            let mut sp = 0.0;
            for i in 0..ncol {
                sp -= self.fact[icol[i as usize]];
            }
            return sp;
        }
        if ncol == 1 {
            let mut sp = 0.0;
            for i in 0..nrow {
                sp -= self.fact[irow[i as usize]];
            }
            return sp;
        }
        if nrow * ncol == 4 {
            if irow[1] <= icol[1] {
                return -(self.fact[irow[1]] + self.fact[icol[1]] + self.fact[icol[1] - irow[1]]);
            } else {
                return -(self.fact[icol[1]] + self.fact[irow[1]] + self.fact[irow[1] - icol[1]]);
            }
        }

        let nrp1 = nrow + ncol + 2;
        let mut ir = vec![0i64; (nrow * nrp1) as usize];
        let mut ic = vec![0i64; (ncol * nrp1) as usize];
        let mut nrstk = IV(vec![0; (nrp1 + 1) as usize]);
        let mut ncstk = IV(vec![0; (nrp1 + 1) as usize]);
        let mut lstk = IV(vec![0; (nrp1 + 1) as usize]);
        let mut mstk = IV(vec![0; (nrp1 + 1) as usize]);
        let mut nstk = IV(vec![0; (nrp1 + 1) as usize]);
        let mut ystk = FV(vec![0.0; (nrp1 + 1) as usize]);

        let ircol = |istk: i64| (istk - 1) * nrow;
        let iccol = |istk: i64| (istk - 1) * ncol;

        for i in 1..=nrow {
            ir[(ircol(1) + i - 1) as usize] = irow[(nrow - i) as usize];
        }
        for j in 1..=ncol {
            ic[(iccol(1) + j - 1) as usize] = icol[(ncol - j) as usize];
        }

        let mut nro = nrow;
        let mut nco = ncol;
        nrstk[1] = nro;
        ncstk[1] = nco;
        ystk[1] = 0.0;
        let mut y = 0.0;
        let mut istk = 1i64;
        let mut lvar = 1i64;
        let mut amx = 0.0;
        let mut sp = dspt;
        let mut m = 0i64;
        let mut nn = 0i64;

        let mut state = "TOP";
        loop {
            match state {
                "TOP" => {
                    let ir1 = ir[(ircol(istk)) as usize];
                    let ic1 = ic[(iccol(istk)) as usize];
                    if ir1 > ic1 {
                        if nro >= nco {
                            m = nco - 1;
                            nn = 2;
                        } else {
                            m = nro;
                            nn = 1;
                        }
                    } else if ir1 < ic1 {
                        if nro <= nco {
                            m = nro - 1;
                            nn = 1;
                        } else {
                            m = nco;
                            nn = 2;
                        }
                    } else if nro <= nco {
                        m = nro - 1;
                        nn = 1;
                    } else {
                        m = nco - 1;
                        nn = 2;
                    }
                    state = "L60";
                    continue;
                }
                "L60" => {
                    let (i, j) = if nn == 1 { (lvar, 1) } else { (1, lvar) };
                    let irt = ir[(ircol(istk) + i - 1) as usize];
                    let ict = ic[(iccol(istk) + j - 1) as usize];
                    y += self.fact[imin(irt, ict)];
                    if irt == ict {
                        nro -= 1;
                        nco -= 1;
                        f11act(&mut ir, ircol(istk), ircol(istk + 1), i, nro);
                        f11act(&mut ic, iccol(istk), iccol(istk + 1), j, nco);
                    } else if irt > ict {
                        nco -= 1;
                        f11act(&mut ic, iccol(istk), iccol(istk + 1), j, nco);
                        f8act(&mut ir, ircol(istk), ircol(istk + 1), irt - ict, i, nro);
                    } else {
                        nro -= 1;
                        f11act(&mut ir, ircol(istk), ircol(istk + 1), i, nro);
                        f8act(&mut ic, iccol(istk), iccol(istk + 1), ict - irt, j, nco);
                    }
                    if nro == 1 {
                        let base = iccol(istk + 1);
                        for kk in 1..=nco {
                            y += self.fact[ic[(base + kk - 1) as usize]];
                        }
                        state = "L90";
                        continue;
                    }
                    if nco == 1 {
                        let base = ircol(istk + 1);
                        for kk in 1..=nro {
                            y += self.fact[ir[(base + kk - 1) as usize]];
                        }
                        state = "L90";
                        continue;
                    }
                    lstk[istk] = lvar;
                    mstk[istk] = m;
                    nstk[istk] = nn;
                    istk += 1;
                    nrstk[istk] = nro;
                    ncstk[istk] = nco;
                    ystk[istk] = y;
                    lvar = 1;
                    state = "TOP";
                    continue;
                }
                "L90" => {
                    if y > amx {
                        amx = y;
                        if sp - amx <= tol {
                            return -dspt;
                        }
                    }
                    state = "L100";
                    continue;
                }
                "L100" => {
                    istk -= 1;
                    if istk == 0 {
                        sp -= amx;
                        if sp - amx <= tol {
                            return -dspt;
                        } else {
                            return sp - dspt;
                        }
                    }
                    lvar = lstk[istk] + 1;
                    state = "L110";
                    continue;
                }
                "L110" => {
                    let mut go60 = false;
                    loop {
                        if lvar > mstk[istk] {
                            break;
                        }
                        nn = nstk[istk];
                        nro = nrstk[istk];
                        nco = ncstk[istk];
                        y = ystk[istk];
                        if nn == 1 {
                            if ir[(ircol(istk) + lvar - 1) as usize]
                                < ir[(ircol(istk) + lvar - 2) as usize]
                            {
                                go60 = true;
                                break;
                            }
                        } else if nn == 2
                            && ic[(iccol(istk) + lvar - 1) as usize]
                                < ic[(iccol(istk) + lvar - 2) as usize]
                        {
                            go60 = true;
                            break;
                        }
                        lvar += 1;
                    }
                    if go60 {
                        state = "L60";
                    } else {
                        state = "L100";
                    }
                    continue;
                }
                _ => unreachable!(),
            }
        }
    }

    fn f5xact(
        &mut self,
        pastp: f64,
        kval: i64,
        ifreq: i64,
        mut itop: i64,
        jkey: i64,
        jstp: i64,
        jstp2: i64,
        jstp3: i64,
        jstp4: i64,
        psh: bool,
    ) -> Result<i64, String> {
        let ldkey = self.ldkey;
        let ldstp = self.ldstp;
        let tol = TOL;

        if psh {
            let ird = kval % ldkey;
            let mut target = 0u8; // 0 none, 30, 40
            let mut itp = -1i64;
            let mut t = ird;
            while t < ldkey {
                if self.key[jkey + t] == kval {
                    itp = t;
                    target = 40;
                    break;
                }
                if self.key[jkey + t] < 0 {
                    itp = t;
                    target = 30;
                    break;
                }
                t += 1;
            }
            if target == 0 {
                let mut t = 0;
                while t < ird {
                    if self.key[jkey + t] == kval {
                        itp = t;
                        target = 40;
                        break;
                    }
                    if self.key[jkey + t] < 0 {
                        itp = t;
                        target = 30;
                        break;
                    }
                    t += 1;
                }
            }
            if target == 0 {
                return Err(format!(
                    "FEXACT error 6 (f5xact): LDKEY={} too small (kval={}); increase 'workspace'",
                    ldkey, kval
                ));
            }
            if target == 30 {
                self.key[jkey + itp] = kval;
                itop += 1;
                self.ipoin[jkey + itp] = itop;
                if itop > ldstp {
                    return Err(format!(
                        "FEXACT error 7 (f5xact): LDSTP={} too small; increase 'workspace' \
                         (or use simulate_p_value=True)",
                        ldstp
                    ));
                }
                self.ifrq[jstp2 + itop - 1] = -1;
                self.ifrq[jstp3 + itop - 1] = -1;
                self.ifrq[jstp4 + itop - 1] = -1;
                self.stp[jstp + itop - 1] = pastp;
                self.ifrq[jstp + itop - 1] = ifreq;
                self.f5_itp = itp;
                return Ok(itop);
            }
            self.f5_itp = itp;
        }

        let itp = self.f5_itp;
        let mut ipn = self.ipoin[jkey + itp];
        let test1 = pastp - tol;
        let test2 = pastp + tol;
        loop {
            let s = self.stp[jstp + ipn - 1];
            if s < test1 {
                ipn = self.ifrq[jstp4 + ipn - 1];
            } else if s > test2 {
                ipn = self.ifrq[jstp3 + ipn - 1];
            } else {
                if INT_MAX - self.ifrq[jstp + ipn - 1] < ifreq {
                    return Err("integer overflow in exact computation".into());
                }
                self.ifrq[jstp + ipn - 1] += ifreq;
                return Ok(itop);
            }
            if !(ipn > 0) {
                break;
            }
        }

        itop += 1;
        if itop > ldstp {
            return Err(format!(
                "FEXACT error 7 (f5xact): LDSTP={} too small; increase 'workspace' \
                 (or use simulate_p_value=True)",
                ldstp
            ));
        }
        let mut ipn = self.ipoin[jkey + itp];
        let mut itmp = ipn;
        loop {
            let s = self.stp[jstp + ipn - 1];
            if s < test1 {
                itmp = ipn;
                ipn = self.ifrq[jstp4 + ipn - 1];
                if ipn > 0 {
                    continue;
                }
                self.ifrq[jstp4 + itmp - 1] = itop;
                break;
            } else if s > test2 {
                itmp = ipn;
                ipn = self.ifrq[jstp3 + ipn - 1];
                if ipn > 0 {
                    continue;
                }
                self.ifrq[jstp3 + itmp - 1] = itop;
                break;
            } else {
                break;
            }
        }
        self.ifrq[jstp2 + itop - 1] = self.ifrq[jstp2 + itmp - 1];
        self.ifrq[jstp2 + itmp - 1] = itop;
        self.stp[jstp + itop - 1] = pastp;
        self.ifrq[jstp + itop - 1] = ifreq;
        self.ifrq[jstp4 + itop - 1] = -1;
        self.ifrq[jstp3 + itop - 1] = -1;
        Ok(itop)
    }

    fn f6xact(&mut self, nrow: i64, ikkey: i64, ldkey: i64, mut last: i64) -> (bool, i64, i64) {
        loop {
            last += 1;
            if last <= ldkey {
                if self.key[ikkey + last] < 0 {
                    continue;
                }
                let mut kval = self.key[ikkey + last];
                self.key[ikkey + last] = -9999;
                let mut j = nrow - 1;
                while j > 0 {
                    self.iro[1 + j] = kval / self.kyy[1 + j];
                    kval -= self.iro[1 + j] * self.kyy[1 + j];
                    j -= 1;
                }
                self.iro[1] = kval;
                return (false, last, last);
            } else {
                return (true, 0, 0);
            }
        }
    }

    fn f7xact(&mut self, nrow: i64, mut k: i64, mut ks: i64) -> (bool, i64, i64) {
        if ks == 0 {
            loop {
                ks += 1;
                if self.idif[ks] != self.iro[ks] {
                    break;
                }
            }
        }
        if self.idif[k] > 0 && k > ks {
            self.idif[k] -= 1;
            loop {
                k -= 1;
                if self.iro[k] != 0 {
                    break;
                }
            }
            let mut m = k;
            while self.idif[m] >= self.iro[m] {
                m -= 1;
            }
            self.idif[m] += 1;
            if m == ks && self.idif[m] == self.iro[m] {
                ks = k;
            }
            return (true, k, ks);
        }

        loop {
            // Loop
            let mut kk = k + 1;
            let mut found_l70 = false;
            while kk <= nrow {
                if self.idif[kk] > 0 {
                    found_l70 = true;
                    break;
                }
                kk += 1;
            }
            if !found_l70 {
                return (false, k, ks);
            }
            // L70
            let mut mm = 1i64;
            for i in 1..=k {
                mm += self.idif[i];
                self.idif[i] = 0;
            }
            k = kk;
            loop {
                k -= 1;
                let m = imin(mm, self.iro[k]);
                self.idif[k] = m;
                mm -= m;
                if !(mm > 0 && k != 1) {
                    break;
                }
            }
            if mm > 0 {
                if kk != nrow {
                    k = kk;
                    continue;
                }
                return (false, k, ks);
            }
            self.idif[kk] -= 1;
            ks = 0;
            loop {
                ks += 1;
                if ks > k {
                    return (true, k, ks);
                }
                if !(self.idif[ks] >= self.iro[ks]) {
                    break;
                }
            }
            return (true, k, ks);
        }
    }
}

/// R's `fexact()` — Fisher's exact test p-value ("PRE") for the `nrow`×`ncol`
/// contingency `table` (row-major flat, length `nrow*ncol`). `expect=-1,
/// percnt=100, emin=0` requests the exact p-value; `expect>0` the hybrid.
#[pyfunction]
#[pyo3(signature = (nrow, ncol, table, expect=-1.0, percnt=100.0, emin=0.0, workspace=200000, mult=30))]
#[allow(clippy::too_many_arguments)]
fn fexact(
    nrow: i64,
    ncol: i64,
    table: Vec<i64>,
    expect: f64,
    percnt: f64,
    emin: f64,
    workspace: i64,
    mult: i64,
) -> PyResult<f64> {
    let mut inst = Fexact {
        nrow,
        ncol,
        m: table,
        expect,
        percnt,
        emin,
        workspace,
        mult,
        ldkey: 0,
        ldstp: 0,
        n2_stack: 0,
        iro: IV(Vec::new()),
        ico: IV(Vec::new()),
        kyy: IV(Vec::new()),
        idif: IV(Vec::new()),
        irn: IV(Vec::new()),
        key: IV(Vec::new()),
        key2: IV(Vec::new()),
        ipoin: IV(Vec::new()),
        lp: FV(Vec::new()),
        sp: FV(Vec::new()),
        tm: FV(Vec::new()),
        stp: FV(Vec::new()),
        ifrq: IV(Vec::new()),
        fact: FV(Vec::new()),
        f3_nst: 0,
        f3_nitc: 0,
        f5_itp: 0,
    };
    inst.run().map_err(PyRuntimeError::new_err)
}

/// Register the FEXACT kernel onto the `_rs` module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fexact, m)?)?;
    Ok(())
}
