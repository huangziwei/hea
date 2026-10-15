use super::Idx;

/// `ikv_t` — `gklib_defs.h:25`, `GK_MKKEYVALUE_T (ikv_t, idx_t, idx_t)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ikv {
    pub key: Idx,
    pub val: Idx,
}

/// `_GKQSORT_MAX_THRESH` (`gk_mksort.h:107`) — "chosen to work best on a Sun
/// 4/260", and load-bearing: it is the partition size below which quicksort
/// stops and the final insertion sort takes over.
const MAX_THRESH: isize = 4;

/// `GK_MKQSORT` (`gk_mksort.h:118-262`).
///
/// Ported with `isize` indices rather than pointers because two of upstream's
/// walks legitimately step one element outside the array — `--_lo >= _tmp_ptr`
/// in the shift loop, and `while (LT (_run_ptr, _tmp_ptr)) --_tmp_ptr` in the
/// insertion sort, which is bounded only by the invariant that the smallest
/// element was already moved to `_base`.
pub fn gk_mkqsort<T: Copy, F: Fn(&T, &T) -> bool>(base: &mut [T], nelt: usize, lt: F) {
    let elems = nelt as isize;
    if elems == 0 {
        return;
    }

    if elems > MAX_THRESH {
        let mut stack = [(0isize, 0isize); 8 * std::mem::size_of::<usize>()];
        let mut top = 1usize;
        let mut lo = 0isize;
        let mut hi = elems - 1;

        while top > 0 {
            let mut mid = lo + ((hi - lo) >> 1);

            if lt(&base[mid as usize], &base[lo as usize]) {
                base.swap(mid as usize, lo as usize);
            }
            if lt(&base[hi as usize], &base[mid as usize]) {
                base.swap(mid as usize, hi as usize);
                if lt(&base[mid as usize], &base[lo as usize]) {
                    base.swap(mid as usize, lo as usize);
                }
            }

            let mut left = lo + 1;
            let mut right = hi - 1;

            loop {
                while lt(&base[left as usize], &base[mid as usize]) {
                    left += 1;
                }
                while lt(&base[mid as usize], &base[right as usize]) {
                    right -= 1;
                }

                if left < right {
                    base.swap(left as usize, right as usize);
                    if mid == left {
                        mid = right;
                    } else if mid == right {
                        mid = left;
                    }
                    left += 1;
                    right -= 1;
                } else if left == right {
                    left += 1;
                    right -= 1;
                    break;
                }
                if left > right {
                    break;
                }
            }

            if right - lo <= MAX_THRESH {
                if hi - left <= MAX_THRESH {
                    top -= 1;
                    (lo, hi) = stack[top];
                } else {
                    lo = left;
                }
            } else if hi - left <= MAX_THRESH {
                hi = right;
            } else if right - lo > hi - left {
                stack[top] = (lo, right);
                top += 1;
                lo = left;
            } else {
                stack[top] = (left, hi);
                top += 1;
                hi = right;
            }
        }
    }

    let end = elems - 1;
    let mut tmp = 0isize;
    let thresh = MAX_THRESH.min(end);

    let mut run = 1isize;
    while run <= thresh {
        if lt(&base[run as usize], &base[tmp as usize]) {
            tmp = run;
        }
        run += 1;
    }
    if tmp != 0 {
        base.swap(tmp as usize, 0);
    }

    run = 2;
    while run <= end {
        tmp = run - 1;
        while lt(&base[run as usize], &base[tmp as usize]) {
            tmp -= 1;
        }
        tmp += 1;

        if tmp != run {
            let hold = base[run as usize];
            let mut h = run;
            let mut l = run;
            loop {
                l -= 1;
                if l < tmp {
                    break;
                }
                base[h as usize] = base[l as usize];
                h = l;
            }
            base[h as usize] = hold;
        }
        run += 1;
    }
}

/// `ikvsorti` (`gklib.c:78-82`) — `ikv_t` by key, ascending.
pub fn ikvsorti(n: usize, base: &mut [Ikv]) {
    gk_mkqsort(base, n, |a, b| a.key < b.key);
}

/// `ikvsortii` (`gklib.c:85-90`) — by key then val.
#[allow(dead_code)]
pub fn ikvsortii(n: usize, base: &mut [Ikv]) {
    gk_mkqsort(base, n, |a, b| {
        a.key < b.key || (a.key == b.key && a.val < b.val)
    });
}

/// `ikvsortd` (`gklib.c:92-97`) — by key, descending.
#[allow(dead_code)]
pub fn ikvsortd(n: usize, base: &mut [Ikv]) {
    gk_mkqsort(base, n, |a, b| a.key > b.key);
}

/// `isum` (`gk_mkblas.h:115-124`), always at `incx == 1` here.
pub fn isum(n: usize, x: &[Idx]) -> Idx {
    let mut sum = 0;
    for xi in x.iter().take(n) {
        sum += *xi;
    }
    sum
}

/// `MAKECSR (i, n, a)` (`gk_macros.h:73-78`) — prefix-sum counts in place, then
/// shift right so `a[0] == 0`. `a` must have room for `n + 1`.
pub fn makecsr(n: usize, a: &mut [Idx]) {
    for i in 1..n {
        a[i] += a[i - 1];
    }
    for i in (1..=n).rev() {
        a[i] = a[i - 1];
    }
    a[0] = 0;
}

/// `SHIFTCSR (i, n, a)` (`gk_macros.h:80-84`).
pub fn shiftcsr(n: usize, a: &mut [Idx]) {
    for i in (1..=n).rev() {
        a[i] = a[i - 1];
    }
    a[0] = 0;
}
