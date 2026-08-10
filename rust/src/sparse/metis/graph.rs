//! `libmetis/graph.c` and the `graph_t` of `libmetis/struct.h`.
//!
//! Two representation choices, both forced by Rust and neither observable:
//!
//! **Every array is owned.** `graph_t` carries `free_xadj`/`free_vwgt`/… flags
//! because `SetupGraph` may borrow the caller's arrays. The only borrowing call
//! site is `METIS_NodeND`'s own fallback when neither pruning nor compression
//! happened, and it never writes them — except in `MMDOrder`, which shifts
//! `xadj`/`adjncy` to 1-based and back. Owning throughout costs one copy of the
//! input on that path and removes the flags entirely.
//!
//! **The `coarser`/`finer` chain is a `Vec<Graph>` stack, not two pointers.**
//! Upstream links each graph to the next coarser one and walks back up through
//! `finer` during uncoarsening. The chain is only ever a stack — `CoarsenGraph`
//! pushes, `Refine2WayNode` pops — so the port keeps it as one, and `graph` is
//! an index into it: `levels[base]` is the C's `graph` and `levels[base + 1]`
//! is `graph->coarser`. That is what lets `Project2WayNodePartition` hold both
//! at once.

use super::super::ws::Ws;
use super::gklib::isum;
use super::{Idx, Real};

/// `nrinfo_t` (`struct.h:76-78`) — a separator vertex's weight on each side.
#[derive(Clone, Copy, Debug, Default)]
pub struct NrInfo {
    pub edegrees: [Idx; 2],
}

/// `graph_t` (`struct.h:81-118`), restricted to what `METIS_NodeND` reads.
///
/// `vsize`, `ckrinfo` and `vkrinfo` belong to the volume and k-way objectives
/// and are not carried.
#[derive(Default)]
pub struct Graph {
    pub nvtxs: Idx,
    pub nedges: Idx,
    pub ncon: Idx,

    pub xadj: Vec<Idx>,
    pub vwgt: Vec<Idx>,
    pub adjncy: Vec<Idx>,
    pub adjwgt: Vec<Idx>,

    pub tvwgt: Vec<Idx>,
    pub invtvwgt: Vec<Real>,

    pub label: Vec<Idx>,
    pub cmap: Vec<Idx>,

    // Partition parameters
    pub mincut: Idx,
    pub r#where: Vec<Idx>,
    pub pwgts: Vec<Idx>,
    pub nbnd: Idx,
    pub bndptr: Vec<Idx>,
    pub bndind: Vec<Idx>,

    // Bisection refinement parameters
    pub id: Vec<Idx>,
    pub ed: Vec<Idx>,

    // Node refinement information
    pub nrinfo: Vec<NrInfo>,
}

impl Graph {
    /// `CreateGraph` / `InitGraph` (`graph.c:158-180`) — all zeros.
    pub fn new() -> Graph {
        Graph::default()
    }

    /// `SetupGraph` (`graph.c:17-90`), for `ncon == 1` and
    /// `objtype != METIS_OBJTYPE_VOL`, which is what OMETIS always is.
    ///
    /// `optype` is OMETIS here, so `SetupGraph_label` always runs.
    pub fn setup(
        nvtxs: Idx,
        ncon: Idx,
        xadj: &[Idx],
        adjncy: &[Idx],
        vwgt: Option<&[Idx]>,
    ) -> Graph {
        let mut graph = Graph::new();

        graph.nvtxs = nvtxs;
        graph.nedges = xadj[nvtxs as usize];
        graph.ncon = ncon;

        graph.xadj = xadj[..=nvtxs as usize].to_vec();
        graph.adjncy = adjncy[..graph.nedges as usize].to_vec();

        graph.vwgt = match vwgt {
            Some(v) => v[..(ncon * nvtxs) as usize].to_vec(),
            None => vec![1; (ncon * nvtxs) as usize],
        };

        graph.tvwgt = vec![0; ncon as usize];
        graph.invtvwgt = vec![0.0; ncon as usize];
        for i in 0..ncon as usize {
            graph.tvwgt[i] = isum_strided(nvtxs, &graph.vwgt[i..], ncon);
            graph.invtvwgt[i] = invtvwgt_of(graph.tvwgt[i]);
        }

        // edge-cut objective: unit edge weights
        graph.adjwgt = vec![1; graph.nedges as usize];

        graph.setup_tvwgt();
        graph.setup_label();

        graph
    }

    /// `SetupGraph_tvwgt` (`graph.c:96-109`).
    pub fn setup_tvwgt(&mut self) {
        if self.tvwgt.is_empty() {
            self.tvwgt = vec![0; self.ncon as usize];
        }
        if self.invtvwgt.is_empty() {
            self.invtvwgt = vec![0.0; self.ncon as usize];
        }
        for i in 0..self.ncon as usize {
            self.tvwgt[i] = isum_strided(self.nvtxs, &self.vwgt[i..], self.ncon);
            self.invtvwgt[i] = invtvwgt_of(self.tvwgt[i]);
        }
    }

    /// `SetupGraph_label` (`graph.c:115-124`).
    pub fn setup_label(&mut self) {
        if self.label.is_empty() {
            self.label = vec![0; self.nvtxs as usize];
        }
        for i in 0..self.nvtxs as usize {
            self.label[i] = i as Idx;
        }
    }

    /// `SetupSplitGraph` (`graph.c:130-152`).
    ///
    /// Upstream leaves the arrays uninitialized; here they are zeroed, which is
    /// unobservable — `SplitGraphOrder` overwrites every entry it later reads,
    /// and the allocator-residue check in `dev/sparse_gates/metis` confirms it.
    pub fn setup_split(&self, snvtxs: Idx, snedges: Idx) -> Graph {
        let mut s = Graph::new();
        s.nvtxs = snvtxs;
        s.nedges = snedges;
        s.ncon = self.ncon;

        s.xadj = vec![0; (snvtxs + 1) as usize];
        s.vwgt = vec![0; (s.ncon * snvtxs) as usize];
        s.adjncy = vec![0; snedges as usize];
        s.adjwgt = vec![0; snedges as usize];
        s.label = vec![0; snvtxs as usize];
        s.tvwgt = vec![0; s.ncon as usize];
        s.invtvwgt = vec![0.0; s.ncon as usize];
        s
    }

    /// `FreeRData` (`graph.c:...`) — drops everything a refinement run built,
    /// so the next of `nseps` runs starts clean. `cmap` and `label` survive.
    pub fn free_rdata(&mut self) {
        self.r#where = Vec::new();
        self.pwgts = Vec::new();
        self.id = Vec::new();
        self.ed = Vec::new();
        self.bndptr = Vec::new();
        self.bndind = Vec::new();
        self.nrinfo = Vec::new();
    }
}

/// `ListInsert` / `BNDInsert` (`macros.h:46-51`, `:65-66`).
///
/// The arrays arrive as [`Ws`] because every caller has already taken its
/// prologue through it. The `debug_assert` is upstream's
/// `ASSERT (lptr[i] == -1)`, and it is what makes the elided bound honest —
/// see `sparse::ws`.
#[inline]
pub fn bnd_insert(nbnd: &mut Idx, bndind: &mut Ws, bndptr: &mut Ws, i: usize) {
    debug_assert_eq!(bndptr[i], -1);
    bndind[*nbnd] = i as Idx;
    bndptr[i] = *nbnd;
    *nbnd += 1;
}

/// `ListDelete` / `BNDDelete` (`macros.h:53-59`, `:68-69`) — swap the last
/// boundary entry into the hole, so `bndind` stays dense and unordered.
#[inline]
pub fn bnd_delete(nbnd: &mut Idx, bndind: &mut Ws, bndptr: &mut Ws, i: usize) {
    debug_assert_ne!(bndptr[i], -1);
    *nbnd -= 1;
    bndind[bndptr[i]] = bndind[*nbnd];
    bndptr[bndind[*nbnd]] = bndptr[i];
    bndptr[i] = -1;
}

/// `1.0/(tvwgt > 0 ? tvwgt : 1)` (`graph.c:50`, `graph.c:107`) — a `double`
/// reciprocal rounded once into `real_t` on the store.
#[inline]
fn invtvwgt_of(tvwgt: Idx) -> Real {
    let d = if tvwgt > 0 { tvwgt } else { 1 };
    (1.0f64 / d as f64) as Real
}

/// `isum (nvtxs, vwgt + i, ncon)` — the strided form `SetupGraph` uses to sum
/// one constraint's weights out of an interleaved `ncon`-wide array.
pub(super) fn isum_strided(n: Idx, x: &[Idx], incx: Idx) -> Idx {
    if incx == 1 {
        return isum(n as usize, x);
    }
    let mut sum = 0;
    for i in 0..n as usize {
        sum += x[i * incx as usize];
    }
    sum
}
