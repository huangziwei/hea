//! The corpus that keeps `METIS_NodeND` honest inside `cargo test`.
//!
//! Two jobs, and the second is the load-bearing one:
//!
//! 1. **Correctness.** Each graph's `perm` and `iperm` are pinned as FNV-1a
//!    digests taken from `dev/sparse_gates/metis`'s oracle — upstream METIS
//!    5.1.0 compiled through `cholmod_metis_wrapper.c`. The private gates check
//!    the same thing over a much larger corpus; these are what ships, so a
//!    change that moves any ordering fails here without a C toolchain.
//! 2. **Bounds.** The kernels below index through [`super::Ws`], which elides
//!    its check outside `debug_assertions`. `cargo test` is a debug build, so
//!    every access this corpus makes is checked — which is the only thing that
//!    makes the release build's elision honest, and the discipline
//!    `sparse::ws` sets out.
//!
//! The shapes are chosen to reach the branches, not for variety: a chain and a
//! band exercise the compression path (both compress well) and `MMDOrder` on
//! the leaves, the 3D grids force real separators and several coarsening
//! levels, `arrow-200` is a single dense row (one vertex adjacent to all), and
//! the random SPD matrices reach `Match_2Hop` and the mask-free contraction.

use super::{metis_nodend, Idx};

/// FNV-1a over the permutation's little-endian `i64` bytes.
fn fnv(a: &[Idx]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in a {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// A 2D 5-point Laplacian pattern on an `m x m` grid, as the symmetric
/// adjacency `METIS_NodeND` takes: both halves, no diagonal.
fn grid2d(m: usize) -> (Idx, Vec<Idx>, Vec<Idx>) {
    graph(m * m, |v, push| {
        let (i, j) = (v / m, v % m);
        if i > 0 {
            push(v - m)
        }
        if j > 0 {
            push(v - 1)
        }
        if j + 1 < m {
            push(v + 1)
        }
        if i + 1 < m {
            push(v + m)
        }
    })
}

/// A 3D 7-point Laplacian pattern on an `m x m x m` grid.
fn grid3d(m: usize) -> (Idx, Vec<Idx>, Vec<Idx>) {
    graph(m * m * m, |v, push| {
        let (k, r) = (v / (m * m), v % (m * m));
        let (i, j) = (r / m, r % m);
        if k > 0 {
            push(v - m * m)
        }
        if i > 0 {
            push(v - m)
        }
        if j > 0 {
            push(v - 1)
        }
        if j + 1 < m {
            push(v + 1)
        }
        if i + 1 < m {
            push(v + m)
        }
        if k + 1 < m {
            push(v + m * m)
        }
    })
}

/// A banded pattern with half-bandwidth `w`.
///
/// Neighbours ascending, like every other generator here: the order *within* a
/// vertex's list is part of the input, because `Match_RM` takes the first
/// unmatched neighbour it sees.
fn band(n: usize, w: usize) -> (Idx, Vec<Idx>, Vec<Idx>) {
    graph(n, |v, push| {
        for d in (1..=w).rev() {
            if v >= d {
                push(v - d)
            }
        }
        for d in 1..=w {
            if v + d < n {
                push(v + d)
            }
        }
    })
}

/// One vertex adjacent to every other, and nothing else — the shape that makes
/// `Match_RM`'s island handling and the dense-row branches run.
fn arrow(n: usize) -> (Idx, Vec<Idx>, Vec<Idx>) {
    graph(n, |v, push| {
        if v == 0 {
            for k in 1..n {
                push(k)
            }
        } else {
            push(0)
        }
    })
}

/// `A @ A.T + shift*I`'s pattern for a sparse random `A`, generated with the
/// same LCG on both sides so the shape is reproducible without a fixture.
fn spd(n: usize, per_col: usize, seed: u64) -> (Idx, Vec<Idx>, Vec<Idx>) {
    let mut state = seed;
    let mut rand = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    // columns of A, then the pattern of A A' is "share a row"
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for col in rows.iter_mut() {
        for _ in 0..per_col {
            col.push(rand() % n);
        }
    }
    let mut adj: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); n];
    let mut byrow: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (c, col) in rows.iter().enumerate() {
        for &r in col {
            byrow[r].push(c);
        }
    }
    for cols in &byrow {
        for &a in cols {
            for &b in cols {
                if a != b {
                    adj[a].insert(b);
                }
            }
        }
    }
    let mut xadj = vec![0 as Idx; n + 1];
    let mut adjncy = Vec::new();
    for v in 0..n {
        for &u in &adj[v] {
            adjncy.push(u as Idx);
        }
        xadj[v + 1] = adjncy.len() as Idx;
    }
    (n as Idx, xadj, adjncy)
}

/// Build CSR/CSC (they coincide, the pattern being symmetric) from a per-vertex
/// neighbour generator.
fn graph(
    n: usize,
    mut nbrs: impl FnMut(usize, &mut dyn FnMut(usize)),
) -> (Idx, Vec<Idx>, Vec<Idx>) {
    let mut xadj = vec![0 as Idx; n + 1];
    let mut adjncy = Vec::new();
    for v in 0..n {
        nbrs(v, &mut |u| adjncy.push(u as Idx));
        xadj[v + 1] = adjncy.len() as Idx;
    }
    (n as Idx, xadj, adjncy)
}

fn corpus() -> Vec<(&'static str, (Idx, Vec<Idx>, Vec<Idx>))> {
    vec![
        ("grid2d-12", grid2d(12)),
        ("grid2d-25", grid2d(25)),
        ("grid3d-6", grid3d(6)),
        ("grid3d-10", grid3d(10)),
        ("band-200-5", band(200, 5)),
        ("chain-300", band(300, 1)),
        ("arrow-200", arrow(200)),
        ("spd-300", spd(300, 6, 3)),
        ("spd-1000", spd(1000, 4, 4)),
    ]
}

/// `(name, n, fnv(perm), fnv(iperm))` from upstream METIS 5.1.0, via
/// `dev/sparse_gates/metis`. Regenerate only against that oracle.
const PINS: &[(&str, Idx, u64, u64)] = &[
    ("grid2d-12", 144, 10606858657682727653, 6667631166029426853),
    ("grid2d-25", 625, 4525716102406086647, 621647128482809903),
    ("grid3d-6", 216, 17463644659732841509, 5382049877827298149),
    ("grid3d-10", 1000, 2056013632698635393, 9641380635990440117),
    (
        "band-200-5",
        200,
        15688968935394507429,
        15553981759965781477,
    ),
    ("chain-300", 300, 17240395651378534545, 16931672645117098245),
];

#[test]
fn the_ordering_is_the_one_upstream_metis_computes() {
    for &(name, n, hp, hip) in PINS {
        let (_, xadj, adjncy) = corpus()
            .into_iter()
            .find(|(g, _)| *g == name)
            .map(|(_, g)| g)
            .expect("pinned graph is in the corpus");
        let (perm, iperm) =
            metis_nodend(n, &xadj, &adjncy).expect("SetupCtrl accepts the defaults");
        assert_eq!(fnv(&perm), hp, "{name}: perm");
        assert_eq!(fnv(&iperm), hip, "{name}: iperm");
    }
}

/// Whatever it returns must at least be a permutation, and `iperm` its inverse.
/// Runs over the whole corpus, including the shapes with no pin, because the
/// point of the wider sweep is to walk every `Ws` access under
/// `debug_assertions`.
#[test]
fn nodend_never_indexes_out_of_bounds() {
    for (name, (n, xadj, adjncy)) in corpus() {
        let (perm, iperm) =
            metis_nodend(n, &xadj, &adjncy).expect("SetupCtrl accepts the defaults");
        let mut seen = vec![false; n as usize];
        for &p in &perm {
            assert!(p >= 0 && p < n, "{name}: perm out of range");
            assert!(!seen[p as usize], "{name}: perm repeats {p}");
            seen[p as usize] = true;
        }
        for (i, &p) in perm.iter().enumerate() {
            assert_eq!(
                iperm[p as usize], i as Idx,
                "{name}: iperm is not perm's inverse"
            );
        }
    }
}

/// The generator is re-seeded by `SetupCtrl`, so two calls in one process must
/// agree — the property the whole port's reproducibility rests on.
#[test]
fn the_ordering_does_not_depend_on_what_ran_before_it() {
    let (n, xadj, adjncy) = grid3d(6);
    let (other_n, other_x, other_a) = grid2d(25);
    let first = metis_nodend(n, &xadj, &adjncy).unwrap();
    metis_nodend(other_n, &other_x, &other_a).unwrap();
    let second = metis_nodend(n, &xadj, &adjncy).unwrap();
    assert_eq!(first.0, second.0);
    assert_eq!(first.1, second.1);
}
