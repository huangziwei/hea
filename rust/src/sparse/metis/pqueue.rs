//! `GKlib/gk_mkpqueue.h` — the locator-indexed binary heap METIS refines with.
//!
//! `gklib.c:32-33` instantiates it twice, and both instantiations pass
//! `key_gt` as the template's `KEY_LT`:
//!
//! ```text
//! #define key_gt(a, b) ((a) > (b))
//! GK_MKPQUEUE (ipq, ipq_t, ikv_t, idx_t,  idx_t, ikvmalloc, IDX_MAX,  key_gt)
//! GK_MKPQUEUE (rpq, rpq_t, rkv_t, real_t, idx_t, rkvmalloc, REAL_MAX, key_gt)
//! ```
//!
//! So the "priority queue" is a **max**-heap: `GetTop` returns the largest key,
//! and everything the template calls "filter up" moves larger keys towards the
//! root. The parameter is kept as a comparator here rather than hard-coded, so
//! the direction stays visible at the instantiation the way it is upstream.
//!
//! `locator[node]` is the node's position in `heap`, or `-1` when absent — the
//! same sentinel discipline as `bndptr`.
//!
//! Each method opens by destructuring `self`, which is what lets the heap and
//! the locator go through [`Ws`] independently. Every subscript is either a
//! heap position the queue maintains or a `locator` entry it wrote, and the
//! refinement kernels drive this heap once per boundary vertex per pass, so the
//! checked access is worth eliding — under the `debug_assert` discipline
//! `sparse::ws` sets out and `metis::tests` walks.

use super::super::ws::Ws;
use super::{Idx, Real};

/// One `(key, val)` heap entry — `rkv_t`/`ikv_t` depending on the
/// instantiation.
#[derive(Clone, Copy, Debug)]
struct Kv<K> {
    key: K,
    val: Idx,
}

/// `GK_MKPQUEUE_T (PQT, KVT)` (`gk_struct.h`) plus its methods.
///
/// `LT` is the template's `KEY_LT` parameter.
pub struct PQueue<K: Copy, LT: Fn(K, K) -> bool> {
    nnodes: usize,
    heap: Vec<Kv<K>>,
    locator: Vec<Idx>,
    lt: LT,
}

impl<K: Copy + Default, LT: Fn(K, K) -> bool> PQueue<K, LT> {
    /// `FPRFX ## Create` / `FPRFX ## Init` (`gk_mkpqueue.h:19-39`).
    pub fn new(maxnodes: usize, lt: LT) -> Self {
        PQueue {
            nnodes: 0,
            heap: vec![
                Kv {
                    key: K::default(),
                    val: 0
                };
                maxnodes
            ],
            locator: vec![-1; maxnodes],
            lt,
        }
    }

    /// `FPRFX ## Reset` (`gk_mkpqueue.h:45-54`).
    pub fn reset(&mut self) {
        let PQueue {
            nnodes,
            heap,
            locator,
            ..
        } = self;
        let heap = Ws::new(heap);
        let locator = Ws::new(locator);
        for i in (0..*nnodes).rev() {
            locator[heap[i].val] = -1;
        }
        *nnodes = 0;
    }

    /// `FPRFX ## Length` (`gk_mkpqueue.h:82-85`). Unused by `METIS_NodeND`;
    /// kept because the gate's opcode replay exercises it.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.nnodes
    }

    /// `FPRFX ## Insert` (`gk_mkpqueue.h:91-119`).
    pub fn insert(&mut self, node: Idx, key: K) {
        let PQueue {
            nnodes,
            heap,
            locator,
            lt,
        } = self;
        let heap = Ws::new(heap);
        let locator = Ws::new(locator);
        let mut i = *nnodes;
        *nnodes += 1;
        while i > 0 {
            let j = (i - 1) >> 1;
            if lt(key, heap[j].key) {
                heap[i] = heap[j];
                locator[heap[i].val] = i as Idx;
                i = j;
            } else {
                break;
            }
        }
        heap[i].key = key;
        heap[i].val = node;
        locator[node] = i as Idx;
    }

    /// `FPRFX ## Delete` (`gk_mkpqueue.h:125-185`).
    pub fn delete(&mut self, node: Idx) {
        let mut node = node;
        let mut i = self.locator[node as usize] as usize;
        self.locator[node as usize] = -1;

        self.nnodes -= 1;
        if self.nnodes > 0 && self.heap[self.nnodes].val != node {
            node = self.heap[self.nnodes].val;
            let newkey = self.heap[self.nnodes].key;
            let oldkey = self.heap[i].key;

            i = if (self.lt)(newkey, oldkey) {
                self.filter_up(i, newkey)
            } else {
                self.filter_down(i, newkey, self.nnodes)
            };

            let PQueue { heap, locator, .. } = self;
            let heap = Ws::new(heap);
            let locator = Ws::new(locator);
            heap[i].key = newkey;
            heap[i].val = node;
            locator[node] = i as Idx;
        }
    }

    /// `FPRFX ## Update` (`gk_mkpqueue.h:191-247`).
    pub fn update(&mut self, node: Idx, newkey: K) {
        let mut i = self.locator[node as usize] as usize;
        let oldkey = self.heap[i].key;

        i = if (self.lt)(newkey, oldkey) {
            self.filter_up(i, newkey)
        } else {
            self.filter_down(i, newkey, self.nnodes)
        };

        let PQueue { heap, locator, .. } = self;
        let heap = Ws::new(heap);
        let locator = Ws::new(locator);
        heap[i].key = newkey;
        heap[i].val = node;
        locator[node] = i as Idx;
    }

    /// `FPRFX ## GetTop` (`gk_mkpqueue.h:255-305`).
    ///
    /// Note this is *not* `Delete (SeeTopVal ())`: the filter-down here re-reads
    /// `queue->nnodes` rather than caching it, which is the same value, but the
    /// two functions are kept separate upstream and are kept separate here.
    pub fn get_top(&mut self) -> Idx {
        if self.nnodes == 0 {
            return -1;
        }
        self.nnodes -= 1;

        let vtx = self.heap[0].val;
        self.locator[vtx as usize] = -1;

        let i = self.nnodes;
        if i > 0 {
            let key = self.heap[i].key;
            let node = self.heap[i].val;
            let i = self.filter_down(0, key, self.nnodes);
            let PQueue { heap, locator, .. } = self;
            let heap = Ws::new(heap);
            let locator = Ws::new(locator);
            heap[i].key = key;
            heap[i].val = node;
            locator[node] = i as Idx;
        }
        vtx
    }

    /// `FPRFX ## SeeTopVal` (`gk_mkpqueue.h:311-314`).
    pub fn see_top_val(&self) -> Idx {
        if self.nnodes == 0 {
            -1
        } else {
            self.heap[0].val
        }
    }

    /// The `while (i > 0)` half shared by `Insert`, `Delete` and `Update`.
    fn filter_up(&mut self, mut i: usize, newkey: K) -> usize {
        let PQueue {
            nnodes,
            heap,
            locator,
            lt,
        } = self;
        let heap = Ws::new(heap);
        let locator = Ws::new(locator);
        let _ = nnodes;
        while i > 0 {
            let j = (i - 1) >> 1;
            if lt(newkey, heap[j].key) {
                heap[i] = heap[j];
                locator[heap[i].val] = i as Idx;
                i = j;
            } else {
                break;
            }
        }
        i
    }

    /// The `while ((j = (i << 1) + 1) < nnodes)` half shared by `Delete`,
    /// `Update` and `GetTop`.
    fn filter_down(&mut self, mut i: usize, newkey: K, nnodes: usize) -> usize {
        let PQueue {
            heap, locator, lt, ..
        } = self;
        let heap = Ws::new(heap);
        let locator = Ws::new(locator);
        loop {
            let mut j = (i << 1) + 1;
            if j >= nnodes {
                break;
            }
            if lt(heap[j].key, newkey) {
                if j + 1 < nnodes && lt(heap[j + 1].key, heap[j].key) {
                    j += 1;
                }
                heap[i] = heap[j];
                locator[heap[i].val] = i as Idx;
                i = j;
            } else if j + 1 < nnodes && lt(heap[j + 1].key, newkey) {
                j += 1;
                heap[i] = heap[j];
                locator[heap[i].val] = i as Idx;
                i = j;
            } else {
                break;
            }
        }
        i
    }
}

/// The `real_t`-keyed instantiation, the only one `METIS_NodeND` reaches.
pub type RPQueue = PQueue<Real, fn(Real, Real) -> bool>;

/// `rpqCreate (maxnodes)` with `gklib.c:33`'s `key_gt`.
pub fn rpq_create(maxnodes: usize) -> RPQueue {
    PQueue::new(maxnodes, |a: Real, b: Real| a > b)
}
