//! `libmetis/options.c` and the `ctrl_t` of `libmetis/struct.h`.
//!
//! `SetupCtrl` has one arm per entry point; only `METIS_OP_OMETIS` is ported,
//! because `METIS_NodeND` is the only entry point this module has, and it calls
//! `SetupCtrl (METIS_OP_OMETIS, options, 1, 3, NULL, NULL)` (`ometis.c:65`).
//! `cholmod_metis.c:782` passes `options = NULL`, so every `GETOPTION` takes its
//! default — but the option array is carried anyway so the defaults stay
//! visible as defaults.
//!
//! The generator state lives here rather than in a static, because in the C it
//! is a file-scope pair in `random.c` reachable from everything, and `ctrl` is
//! the one thing every `irand*` call site already has.

use super::rng::Rng;
use super::{Idx, Real};

/// `moptype_et` (`metis.h:263-267`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OpType {
    Ometis = 2,
}

/// `mobjtype_et` (`metis.h:357-361`). Only `_NODE` is reachable — `SetupCtrl`'s
/// OMETIS arm defaults to it and `CheckParams` rejects anything else — but the
/// three are kept together because the default is written as a `GETOPTION`.
#[allow(dead_code)]
pub const METIS_OBJTYPE_CUT: Idx = 0;
#[allow(dead_code)]
pub const METIS_OBJTYPE_VOL: Idx = 1;
pub const METIS_OBJTYPE_NODE: Idx = 2;

/// `mctype_et` (`metis.h:313-316`).
pub const METIS_CTYPE_RM: Idx = 0;
pub const METIS_CTYPE_SHEM: Idx = 1;

/// `miptype_et` (`metis.h:319-325`).
#[allow(dead_code)]
pub const METIS_IPTYPE_GROW: Idx = 0;
#[allow(dead_code)]
pub const METIS_IPTYPE_RANDOM: Idx = 1;
pub const METIS_IPTYPE_EDGE: Idx = 2;
pub const METIS_IPTYPE_NODE: Idx = 3;

/// `mrtype_et` (`metis.h:333-338`).
pub const METIS_RTYPE_SEP2SIDED: Idx = 2;
pub const METIS_RTYPE_SEP1SIDED: Idx = 3;

/// `moptions_et` (`metis.h:271-296`) — the indices into the `options` array.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub enum Opt {
    Ptype = 0,
    Objtype = 1,
    Ctype = 2,
    Iptype = 3,
    Rtype = 4,
    Dbglvl = 5,
    Niter = 6,
    Ncuts = 7,
    Seed = 8,
    No2hop = 9,
    Minconn = 10,
    Contig = 11,
    Compress = 12,
    Ccorder = 13,
    Pfactor = 14,
    Nseps = 15,
    Ufactor = 16,
    Numbering = 17,
}

/// `OMETIS_DEFAULT_UFACTOR` (`defs.h:58`).
const OMETIS_DEFAULT_UFACTOR: Idx = 200;

/// `GETOPTION (options, idx, defval)` (`macros.h:31-32`).
#[inline]
fn getoption(options: Option<&[Idx]>, idx: Opt, defval: Idx) -> Idx {
    match options {
        None => defval,
        Some(o) if o[idx as usize] == -1 => defval,
        Some(o) => o[idx as usize],
    }
}

/// `ctrl_t` (`struct.h:139-203`), restricted to what `METIS_NodeND` reads.
///
/// The timers, the k-way neighbour pools and the subdomain graph are all
/// k-way-only and are not carried. `mcore` is not carried either — see
/// [`super::wspace`].
///
/// Three fields are set and never read back: `optype` (the arm `SetupCtrl` was
/// entered through), `dbglvl` (which gates upstream's tracing, not ported) and
/// `tpwgts` (read only by `SetupKWayBalMultipliers`). They stay because
/// `SetupCtrl` writes them and dropping them would make the port of that
/// function partial.
#[allow(dead_code)]
pub struct Ctrl {
    pub optype: OpType,
    pub objtype: Idx,
    pub dbglvl: Idx,
    pub ctype: Idx,
    pub iptype: Idx,
    pub rtype: Idx,

    pub coarsen_to: Idx,
    pub no2hop: Idx,
    pub nseps: Idx,
    pub ufactor: Idx,
    pub compress: Idx,
    pub ccorder: Idx,
    pub seed: Idx,
    pub niter: Idx,
    pub numflag: Idx,
    pub maxvwgt: Vec<Idx>,

    pub ncon: Idx,
    pub nparts: Idx,

    pub pfactor: Real,
    pub ubfactors: Vec<Real>,
    pub tpwgts: Vec<Real>,
    pub pijbm: Vec<Real>,
    pub cfactor: Real,

    pub rng: Rng,
}

impl Ctrl {
    /// `SetupCtrl` (`options.c:17-133`), `METIS_OP_OMETIS` arm.
    ///
    /// Returns `None` where the C returns `NULL` — `CheckParams` rejected the
    /// combination.
    pub fn setup(optype: OpType, options: Option<&[Idx]>, ncon: Idx, nparts: Idx) -> Option<Ctrl> {
        let OpType::Ometis = optype;

        let objtype = getoption(options, Opt::Objtype, METIS_OBJTYPE_NODE);
        let rtype = getoption(options, Opt::Rtype, METIS_RTYPE_SEP1SIDED);
        let iptype = getoption(options, Opt::Iptype, METIS_IPTYPE_EDGE);
        let nseps = getoption(options, Opt::Nseps, 1);
        let niter = getoption(options, Opt::Niter, 10);
        let ufactor = getoption(options, Opt::Ufactor, OMETIS_DEFAULT_UFACTOR);
        let compress = getoption(options, Opt::Compress, 1);
        let ccorder = getoption(options, Opt::Ccorder, 0);
        let pfactor = (0.1 * getoption(options, Opt::Pfactor, 0) as f64) as Real;

        let coarsen_to = 100;

        let ctype = getoption(options, Opt::Ctype, METIS_CTYPE_SHEM);
        let no2hop = getoption(options, Opt::No2hop, 0);
        let seed = getoption(options, Opt::Seed, -1);
        let dbglvl = getoption(options, Opt::Dbglvl, 0);
        let numflag = getoption(options, Opt::Numbering, 0);

        let tpwgts = vec![0.5 as Real; 2];

        let ub = (1.0 + 0.001 * ufactor as f64) as Real;
        let ubfactors = vec![(ub as f64 + 0.0000499) as Real; ncon.max(0) as usize];

        let mut ctrl = Ctrl {
            optype,
            objtype,
            dbglvl,
            ctype,
            iptype,
            rtype,
            coarsen_to,
            no2hop,
            nseps,
            ufactor,
            compress,
            ccorder,
            seed,
            niter,
            numflag,
            maxvwgt: vec![0; ncon.max(0) as usize],
            ncon,
            nparts,
            pfactor,
            ubfactors,
            tpwgts,
            pijbm: vec![0.0; (nparts * ncon).max(0) as usize],
            cfactor: 0.0,
            rng: Rng::default(),
        };

        ctrl.rng.init_random(ctrl.seed);

        if ctrl.check_params() {
            Some(ctrl)
        } else {
            None
        }
    }

    /// `CheckParams` (`options.c:262-...`), `METIS_OP_OMETIS` arm.
    fn check_params(&self) -> bool {
        if self.objtype != METIS_OBJTYPE_NODE {
            return false;
        }
        if self.ctype != METIS_CTYPE_RM && self.ctype != METIS_CTYPE_SHEM {
            return false;
        }
        if self.iptype != METIS_IPTYPE_EDGE && self.iptype != METIS_IPTYPE_NODE {
            return false;
        }
        if self.rtype != METIS_RTYPE_SEP1SIDED && self.rtype != METIS_RTYPE_SEP2SIDED {
            return false;
        }
        if self.nseps <= 0 || self.niter <= 0 || self.ufactor <= 0 {
            return false;
        }
        if self.numflag != 0 && self.numflag != 1 {
            return false;
        }
        if self.nparts != 3 || self.ncon != 1 {
            return false;
        }
        if self.compress != 0 && self.compress != 1 {
            return false;
        }
        if self.ccorder != 0 && self.ccorder != 1 {
            return false;
        }
        if self.pfactor < 0.0 {
            return false;
        }
        for i in 0..self.ncon as usize {
            if self.ubfactors[i] <= 1.0 {
                return false;
            }
        }
        true
    }

    /// `Setup2WayBalMultipliers` (`options.c:154-163`).
    pub fn setup_2way_bal_multipliers(&mut self, invtvwgt: &[Real], ncon: Idx, tpwgts: &[Real]) {
        for i in 0..2usize {
            for j in 0..ncon as usize {
                self.pijbm[i * ncon as usize + j] = invtvwgt[j] / tpwgts[i * ncon as usize + j];
            }
        }
    }
}
