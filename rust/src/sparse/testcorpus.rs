//! Pattern corpus shared by every kernel's `#[cfg(test)]` module.
//!
//! [`Ws`](super::ws::Ws) elides its bounds check outside `debug_assertions`,
//! so the guarantee that these kernels never index out of range rests on a
//! debug build actually walking every subscript they form. `cargo test` is a
//! debug build; the matrices below are chosen so that between them they reach
//! garbage collection, dense-row removal, mass elimination, supervariable
//! detection via hash collisions, multiple etree roots, and the degree-0 path
//! — the branches that compute subscripts in the least obvious ways.
//!
//! Bit-exactness against upstream's C is *not* checked from here. That lives in
//! the Python suite, which pins values taken from SuiteSparse compiled at the
//! target tag. The Rust side checks memory safety and structural invariants;
//! the Python side checks the numbers.

pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    pub fn below(&mut self, hi: usize) -> usize {
        self.next_u32() as usize % hi
    }
}

pub fn triangle_csc(n: usize, edges: &[(usize, usize)], lower: bool) -> (Vec<i64>, Vec<i64>) {
    let mut cols: Vec<Vec<i64>> = vec![Vec::new(); n];
    for (j, col) in cols.iter_mut().enumerate() {
        col.push(j as i64);
    }
    for &(a, b) in edges {
        if a == b || a >= n || b >= n {
            continue;
        }
        let (hi, lo) = (a.max(b), a.min(b));
        /* the stored half puts the entry in the column that keeps it on
         * the correct side of the diagonal */
        if lower {
            cols[lo].push(hi as i64);
        } else {
            cols[hi].push(lo as i64);
        }
    }
    let mut indptr = vec![0i64; n + 1];
    let mut indices = Vec::new();
    for j in 0..n {
        cols[j].sort_unstable();
        cols[j].dedup();
        indices.extend_from_slice(&cols[j]);
        indptr[j + 1] = indices.len() as i64;
    }
    (indptr, indices)
}

pub fn spd_triangle(
    n: usize,
    edges: &[(usize, usize)],
    lower: bool,
) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let (indptr, indices) = triangle_csc(n, edges, lower);
    let mut x = vec![0.0f64; indices.len()];
    let mut rng = Lcg(0xC0FFEE);
    /* off-diagonal magnitudes accumulate into both endpoints' rows */
    let mut absrow = vec![0.0f64; n];
    for j in 0..n {
        for p in indptr[j] as usize..indptr[j + 1] as usize {
            let i = indices[p] as usize;
            if i != j {
                let v = (rng.below(2000) as f64) / 1000.0 - 1.0;
                x[p] = v;
                absrow[i] += v.abs();
                absrow[j] += v.abs();
            }
        }
    }
    for j in 0..n {
        for p in indptr[j] as usize..indptr[j + 1] as usize {
            if indices[p] as usize == j {
                x[p] = absrow[j] + 1.0;
            }
        }
    }
    (indptr, indices, x)
}

pub fn corpus() -> Vec<(&'static str, usize, Vec<(usize, usize)>)> {
    let mut out: Vec<(&'static str, usize, Vec<(usize, usize)>)> = Vec::new();

    out.push(("empty", 0, Vec::new()));
    out.push(("singleton", 1, Vec::new()));
    /* every degree is 0: the all-empty-rows path, and n isolated etree roots */
    out.push(("diagonal-32", 32, Vec::new()));

    for &(n, bw) in &[(50usize, 2usize), (400, 5)] {
        let mut e = Vec::new();
        for j in 0..n {
            for k in 1..=bw {
                if j + k < n {
                    e.push((j, j + k));
                }
            }
        }
        out.push(("banded", n, e));
    }

    /* one row touching everything: the `deg > dense` removal path */
    let mut arrow = Vec::new();
    for j in 1..300 {
        arrow.push((0usize, j));
        if j + 1 < 300 {
            arrow.push((j, j + 1));
        }
    }
    out.push(("arrow-300", 300, arrow));

    /* Sparse random graphs, which is what forces Iw to run out: the elbow
     * room amd_2 gets is proportional to nnz, while the fill-in these
     * produce is not, so the element lists outgrow it and garbage
     * collection runs. Denser random matrices never reach that branch. */
    let mut rng = Lcg(0x5eed);
    for &(n, m) in &[
        (200usize, 400usize),
        (400, 1600),
        (1000, 2000),
        (600, 12000),
    ] {
        let mut e = Vec::with_capacity(m);
        for _ in 0..m {
            e.push((rng.below(n), rng.below(n)));
        }
        out.push(("random", n, e));
    }

    /* rows repeated in blocks of four have identical patterns, which is
     * what drives supervariable detection through its hash buckets */
    let mut dup = Vec::new();
    let (n, blk) = (160usize, 4usize);
    for a in 0..n / blk {
        for b in 0..n / blk {
            if rng.below(3) == 0 {
                for p in 0..blk {
                    for q in 0..blk {
                        dup.push((a * blk + p, b * blk + q));
                    }
                }
            }
        }
    }
    out.push(("duplicate-rows-160", n, dup));

    /* disconnected components exercise the multiple-roots path in postorder */
    let mut blocks = Vec::new();
    let mut base = 0usize;
    for &k in &[7usize, 13, 5, 21, 9] {
        for a in 0..k {
            for b in 0..a {
                if rng.below(2) == 0 {
                    blocks.push((base + a, base + b));
                }
            }
        }
        base += k;
    }
    out.push(("block-diagonal", base, blocks));

    out
}
