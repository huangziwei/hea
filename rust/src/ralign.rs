//! Double buffers laid out where R puts the data of a double vector of the
//! same length: 8 bytes past a 16-byte boundary for lengths 2 to 16 (R's
//! small-vector pools), on a 16-byte boundary otherwise. The vendor `ddot`
//! rounds by the alignment of its second operand, so ports that hand R
//! vectors to the BLAS place their copies the same way.

#[derive(Default)]
pub(crate) struct RVec {
    buf: Vec<f64>,
    off: usize,
    len: usize,
}

impl RVec {
    /// `len` zeros at R's alignment for that length.
    pub(crate) fn zeros(len: usize) -> Self {
        Self::at(len, if (2..=16).contains(&len) { 8 } else { 0 })
    }

    /// `len` zeros on a 16-byte boundary, where C's `calloc` puts them.
    #[cfg(vendor_blas)]
    pub(crate) fn zeros_aligned(len: usize) -> Self {
        Self::at(len, 0)
    }

    /// `len` zeros `want` (0 or 8) bytes past a 16-byte boundary.
    pub(crate) fn at(len: usize, want: usize) -> Self {
        let buf = vec![0.0f64; len + 2];
        let addr = buf.as_ptr() as usize;
        let off = ((want + 16 - addr % 16) % 16) / 8;
        RVec { buf, off, len }
    }

    /// A copy of `s` at R's alignment for its length.
    pub(crate) fn from_slice(s: &[f64]) -> Self {
        let mut v = Self::zeros(s.len());
        v.as_mut_slice().copy_from_slice(s);
        v
    }

    pub(crate) fn as_slice(&self) -> &[f64] {
        &self.buf[self.off..self.off + self.len]
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.buf[self.off..self.off + self.len]
    }
}

/// A copy at the same address mod 16 as the original, standing in for the
/// R memory the original stands for.
impl Clone for RVec {
    fn clone(&self) -> Self {
        let want = (self.as_slice().as_ptr() as usize) % 16;
        let mut v = RVec::at(self.len, want);
        v.as_mut_slice().copy_from_slice(self.as_slice());
        v
    }
}

impl std::ops::Deref for RVec {
    type Target = [f64];
    fn deref(&self) -> &[f64] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for RVec {
    fn deref_mut(&mut self) -> &mut [f64] {
        self.as_mut_slice()
    }
}
