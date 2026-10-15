use crate::nmath::util::rfma;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

const INF: f64 = 1.0e300;

fn hcass2(n: usize, ia: &[i64], ib: &[i64]) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut iorder = vec![0i64; n + 1];
    let mut iia = vec![0i64; n + 1];
    let mut iib = vec![0i64; n + 1];
    for i in 1..=n {
        iia[i] = ia[i];
        iib[i] = ib[i];
    }
    for i in 1..(n - 1) {
        let k = ia[i].min(ib[i]); // smallest (+ve or -ve) seq. no.
        for j in (i + 1)..n {
            if ia[j] == k {
                iia[j] = -(i as i64);
            }
            if ib[j] == k {
                iib[j] = -(i as i64);
            }
        }
    }
    for i in 1..n {
        iia[i] = -iia[i];
        iib[i] = -iib[i];
    }
    for i in 1..n {
        if iia[i] > 0 && iib[i] < 0 {
            std::mem::swap(&mut iia[i], &mut iib[i]);
        }
        if iia[i] > 0 && iib[i] > 0 {
            let k1 = iia[i].min(iib[i]);
            let k2 = iia[i].max(iib[i]);
            iia[i] = k1;
            iib[i] = k2;
        }
    }
    iorder[1] = iia[n - 1];
    iorder[2] = iib[n - 1];
    let mut loc = 2usize;
    for i in (1..=(n - 2)).rev() {
        for j in 1..=loc {
            if iorder[j] == i as i64 {
                iorder[j] = iia[i];
                if j == loc {
                    loc += 1;
                    iorder[loc] = iib[i];
                } else {
                    loc += 1;
                    let mut kk = loc;
                    while kk > j + 1 {
                        iorder[kk] = iorder[kk - 1];
                        kk -= 1;
                    }
                    iorder[j + 1] = iib[i];
                }
                break;
            }
        }
    }
    for i in 1..=n {
        iorder[i] = -iorder[i];
    }
    (iorder, iia, iib)
}

#[pyfunction]
#[pyo3(name = "hclust", signature = (n, diss, iopt, members))]
pub fn hclust<'py>(
    py: Python<'py>,
    n: usize,
    diss: PyReadonlyArray1<'py, f64>,
    iopt: usize,
    members: PyReadonlyArray1<'py, f64>,
) -> (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
) {
    let diss = diss.as_slice().unwrap();
    let memb0 = members.as_slice().unwrap();

    let (iia_out, iib_out, height_out, order_out) = py.allow_threads(|| {
        let length = n * (n - 1) / 2;

        let mut d = vec![0.0f64; length + 1];
        for i in 1..=length {
            d[i] = diss[i - 1];
        }
        let mut ia = vec![0i64; n + 1];
        let mut ib = vec![0i64; n + 1];
        let mut crit = vec![0.0f64; n + 1];
        let mut membr = vec![0.0f64; n + 1];
        for i in 1..=n {
            membr[i] = memb0[i - 1];
        }
        let mut nn = vec![0usize; n + 1];
        let mut disnn = vec![0.0f64; n + 1];
        let mut flag = vec![false; n + 1];

        let ioffst = |i: usize, j: usize| -> usize { j + (i - 1) * n - i * (i + 1) / 2 };

        let (mut im, mut jj, mut jm): (usize, usize, usize) = (0, 0, 0);

        for i in 1..=n {
            flag[i] = true;
        }
        let mut ncl = n;

        let isward = iopt == 1 || iopt == 8;
        if iopt == 8 {
            for i in 1..=length {
                d[i] = d[i] * d[i];
            }
        }

        for i in 1..n {
            let mut dmin = INF;
            let lo = ioffst(i, i + 1);
            for (off, &dv) in d[lo..lo + (n - i)].iter().enumerate() {
                if dmin > dv {
                    dmin = dv;
                    jm = i + 1 + off;
                }
            }
            nn[i] = jm;
            disnn[i] = dmin;
        }

        loop {
            let mut dmin = INF;
            for i in 1..n {
                if flag[i] && disnn[i] < dmin {
                    dmin = disnn[i];
                    im = i;
                    jm = nn[i];
                }
            }
            ncl -= 1;

            let i2 = im.min(jm);
            let j2 = im.max(jm);
            ia[n - ncl] = i2 as i64;
            ib[n - ncl] = j2 as i64;
            if iopt == 8 {
                dmin = dmin.sqrt();
            }
            crit[n - ncl] = dmin;
            flag[j2] = false;

            let d12 = d[ioffst(i2, j2)];
            let mut dmin_nn = INF;
            for k in 1..=n {
                if flag[k] && k != i2 {
                    let ind1 = if i2 < k { ioffst(i2, k) } else { ioffst(k, i2) };
                    let ind2 = if j2 < k { ioffst(j2, k) } else { ioffst(k, j2) };

                    if isward {
                        let v = rfma(
                            membr[i2] + membr[k],
                            d[ind1],
                            (membr[j2] + membr[k]) * d[ind2],
                        );
                        d[ind1] = rfma(-membr[k], d12, v);
                        d[ind1] /= membr[i2] + membr[j2] + membr[k];
                    } else if iopt == 2 {
                        d[ind1] = d[ind1].min(d[ind2]);
                    } else if iopt == 3 {
                        d[ind1] = d[ind1].max(d[ind2]);
                    } else if iopt == 4 {
                        d[ind1] =
                            rfma(membr[i2], d[ind1], membr[j2] * d[ind2]) / (membr[i2] + membr[j2]);
                    } else if iopt == 5 {
                        d[ind1] = (d[ind1] + d[ind2]) / 2.0;
                    } else if iopt == 6 {
                        d[ind1] = ((d[ind1] + d[ind2]) - d12 / 2.0) / 2.0;
                    } else if iopt == 7 {
                        d[ind1] = (rfma(membr[i2], d[ind1], membr[j2] * d[ind2])
                            - membr[i2] * membr[j2] * d12 / (membr[i2] + membr[j2]))
                            / (membr[i2] + membr[j2]);
                    }

                    if i2 < k {
                        if d[ind1] < dmin_nn {
                            dmin_nn = d[ind1];
                            jj = k;
                        }
                    } else if d[ind1] < disnn[k] {
                        disnn[k] = d[ind1];
                        nn[k] = i2;
                    }
                }
            }
            membr[i2] += membr[j2];
            disnn[i2] = dmin_nn;
            nn[i2] = jj;

            for i in 1..n {
                if flag[i] && (nn[i] == i2 || nn[i] == j2) {
                    let mut dmin_r = INF;
                    let lo = ioffst(i, i + 1);
                    for (off, &dv) in d[lo..lo + (n - i)].iter().enumerate() {
                        let j = i + 1 + off;
                        if flag[j] && dv < dmin_r {
                            dmin_r = dv;
                            jj = j;
                        }
                    }
                    nn[i] = jj;
                    disnn[i] = dmin_r;
                }
            }

            if ncl > 1 {
                continue;
            }
            break;
        }

        let (iorder, iia, iib) = hcass2(n, &ia, &ib);
        (
            iia[1..n].to_vec(),     // merge_a, length n-1
            iib[1..n].to_vec(),     // merge_b, length n-1
            crit[1..n].to_vec(),    // height,  length n-1
            iorder[1..=n].to_vec(), // order,   length n
        )
    });

    (
        iia_out.into_pyarray(py),
        iib_out.into_pyarray(py),
        height_out.into_pyarray(py),
        order_out.into_pyarray(py),
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hclust, m)?)?;
    Ok(())
}
