use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "cutree", signature = (merge, which))]
pub fn cutree<'py>(
    py: Python<'py>,
    merge: PyReadonlyArray2<'py, i64>,
    which: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray2<i64>> {
    let m = merge.as_array();
    let n1 = m.shape()[0];
    let n = n1 + 1;
    let which = which.as_slice().unwrap();
    let nw = which.len();
    let col1: Vec<i64> = (0..n1).map(|i| m[[i, 0]]).collect();
    let col2: Vec<i64> = (0..n1).map(|i| m[[i, 1]]).collect();

    let mut ans = vec![0i64; n * nw];
    let mut sing = vec![true; n + 1]; // is k-th obs still alone?
    let mut m_nr = vec![0i64; n + 1]; // last merge step containing k-th obs
    let mut z = vec![0i64; n + 1];

    let mut first_col: usize = 0;
    for k in 1..n {
        let m1_0 = col1[k - 1];
        let m2_0 = col2[k - 1];
        let kk = k as i64;
        if m1_0 < 0 && m2_0 < 0 {
            m_nr[(-m1_0) as usize] = kk;
            m_nr[(-m2_0) as usize] = kk;
            sing[(-m1_0) as usize] = false;
            sing[(-m2_0) as usize] = false;
        } else if m1_0 < 0 || m2_0 < 0 {
            let (j, m1) = if m1_0 < 0 {
                ((-m1_0) as usize, m2_0)
            } else {
                ((-m2_0) as usize, m1_0)
            };
            for v in m_nr[1..=n].iter_mut() {
                if *v == m1 {
                    *v = kk;
                }
            }
            m_nr[j] = kk;
            sing[j] = false;
        } else {
            for v in m_nr[1..=n].iter_mut() {
                if *v == m1_0 || *v == m2_0 {
                    *v = kk;
                }
            }
        }

        let mut found_j = false;
        for j in 0..nw {
            if which[j] == (n - k) as i64 {
                if !found_j {
                    found_j = true;
                    for ell in 1..=n {
                        z[ell] = 0;
                    }
                    let mut nclust = 0i64;
                    first_col = j;
                    for ell in 1..=n {
                        if sing[ell] {
                            nclust += 1;
                            ans[(ell - 1) * nw + j] = nclust;
                        } else {
                            let mnr = m_nr[ell] as usize;
                            if z[mnr] == 0 {
                                nclust += 1;
                                z[mnr] = nclust;
                            }
                            ans[(ell - 1) * nw + j] = z[mnr];
                        }
                    }
                } else {
                    for ell in 0..n {
                        ans[ell * nw + j] = ans[ell * nw + first_col];
                    }
                }
            }
        }
    }
    for j in 0..nw {
        if which[j] == n as i64 {
            for ell in 0..n {
                ans[ell * nw + j] = (ell + 1) as i64;
            }
        }
    }
    Array2::from_shape_vec((n, nw), ans)
        .unwrap()
        .into_pyarray(py)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cutree, m)?)?;
    Ok(())
}
