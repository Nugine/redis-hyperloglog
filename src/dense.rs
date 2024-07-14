use std::alloc::alloc_zeroed;
use std::alloc::dealloc;
use std::alloc::handle_alloc_error;
use std::alloc::Layout;
use std::ptr;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use crate::array::UnsafeArray;
use crate::config::*;

const HLL_BITS_MASK: u16 = (1 << HLL_BITS) - 1;

const DENSE_PAD_LEN: usize = 16;
const DENSE_REGISTERS_LEN: usize = (HLL_REGISTERS * HLL_BITS + 7) / 8 + DENSE_PAD_LEN;

#[repr(C)]
pub struct HllDense {
    repr: HllRepr,
    cmin: u8,
    _pad: [u8; 6],
    card: AtomicU64,
    hist: UnsafeArray<u16, HLL_HIST_LEN>,
    regs: UnsafeArray<u8, DENSE_REGISTERS_LEN>,
}

impl HllDense {
    pub fn create() -> *mut Self {
        let layout = Layout::new::<Self>();
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        let this: *mut Self = ptr.cast();
        unsafe { Self::init_from_zeroed(this) }
        this
    }

    #[allow(clippy::cast_possible_truncation)]
    unsafe fn init_from_zeroed(this: *mut Self) {
        (*this).repr = HllRepr::Dense;
        (*this).cmin = 0;
        (*this).card = const { AtomicU64::new(0) };
        (*this).hist[0] = HLL_REGISTERS as u16;
        // ...zero-initialized
    }

    pub unsafe fn destroy(this: *mut Self) {
        let layout = Layout::new::<Self>();
        dealloc(this.cast(), layout);
    }

    pub fn clear(&mut self) {
        unsafe {
            let this = ptr::from_mut(self);
            this.write_bytes(0, 1);
            Self::init_from_zeroed(this);
        }
    }

    pub fn insert(&mut self, hash: u64) -> bool {
        let (index, count) = hll_pattern(hash);

        if count < self.cmin {
            return false;
        }

        let old_count = unsafe { get_register(self.regs.as_ptr(), index) };

        if count <= old_count {
            return false;
        }

        unsafe {
            set_register(self.regs.as_mut_ptr(), index, count);

            self.hist[old_count] -= 1;
            self.hist[count] += 1;

            if old_count == self.cmin {
                let mut count_min = self.cmin;
                while self.hist[count_min] == 0 {
                    count_min += 1;
                }
                self.cmin = count_min;
            }

            *self.card.get_mut() = u64::MAX;
        }

        true
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_lossless,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    pub fn count(&self) -> u64 {
        let card = self.card.load(Ordering::Relaxed);
        if card != u64::MAX {
            return card;
        }

        let m = HLL_REGISTERS as f64;

        let h_last = self.hist[HLL_Q + 1] as f64;
        let mut z = m * hll_tau((m - h_last) / m);

        let mut i = HLL_Q;
        loop {
            let count = self.hist[i];
            z += count as f64;
            z *= 0.5;

            i -= 1;
            if i == 0 {
                break;
            }
        }

        let h0 = self.hist[0] as f64;
        z += m * hll_sigma(h0 / m);

        let e = HLL_ALPHA_INF * m * m / z;
        let ans = e.round() as u64;

        self.card.store(ans, Ordering::Relaxed);
        ans
    }

    pub fn merge(&mut self, sources: &[&&Self]) {
        unsafe {
            let mut reg_raw = [0; HLL_REGISTERS];

            if *self.card.get_mut() != 0 {
                merge_max(reg_raw.as_mut_ptr(), self.regs.as_ptr());
            }
            for &&src in sources {
                merge_max(reg_raw.as_mut_ptr(), src.regs.as_ptr());
            }

            reg_histogram(self.hist.as_mut_ptr(), reg_raw.as_ptr());

            let mut count_min = 0;
            while self.hist[count_min] == 0 {
                count_min += 1;
            }
            self.cmin = count_min;

            *self.card.get_mut() = u64::MAX;

            compress(self.regs.as_mut_ptr(), reg_raw.as_ptr());
        }
    }
}

#[inline(always)]
#[allow(clippy::cast_possible_truncation, clippy::cast_ptr_alignment)]
unsafe fn get_register(reg_dense: *const u8, index: u32) -> u8 {
    let i = index.unchecked_mul(HLL_BITS as u32) / 8;
    let low = index.unchecked_mul(HLL_BITS as u32) & 7;
    let ptr = reg_dense.add(i as usize).cast::<u16>();
    let b = ptr.read_unaligned();

    ((b.unchecked_shr(low)) & HLL_BITS_MASK) as u8
}

#[inline(always)]
#[allow(clippy::cast_possible_truncation, clippy::cast_ptr_alignment)]
unsafe fn set_register(reg_dense: *mut u8, index: u32, value: u8) {
    let i = index.unchecked_mul(HLL_BITS as u32) / 8;
    let low = index.unchecked_mul(HLL_BITS as u32) & 7;
    let ptr = reg_dense.add(i as usize).cast::<u16>();
    let b = ptr.read_unaligned();
    let value = u16::from(value);

    let mask = u16::MAX ^ (HLL_BITS_MASK.unchecked_shl(low));
    let value = (b & mask) | value.unchecked_shl(low);
    ptr.write_unaligned(value);
}

unsafe fn reg_histogram(hist: *mut u16, reg_raw: *const u8) {
    hist.write_bytes(0, HLL_HIST_LEN);
    for i in 0..HLL_REGISTERS {
        let val = *reg_raw.add(i);
        *hist.add(val as usize) += 1;
    }
}

#[inline(always)]
unsafe fn merge_max(reg_raw: *mut u8, reg_dense: *const u8) {
    if const { HLL_BITS == 6 && HLL_REGISTERS % 64 == 0 }
        && is_simd_enabled()
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
    {
        return merge_max_avx512(reg_raw, reg_dense);
    }
    merge_max_scalar(reg_raw, reg_dense);
}

#[allow(clippy::cast_possible_truncation)]
unsafe fn merge_max_scalar(reg_raw: *mut u8, reg_dense: *const u8) {
    for i in 0..HLL_REGISTERS {
        let val = get_register(reg_dense, i as u32);
        let raw = &mut *reg_raw.add(i);
        if val > *raw {
            *raw = val;
        }
    }
}

#[allow(clippy::many_single_char_names)]
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn merge_max_avx512(reg_raw: *mut u8, reg_dense: *const u8) {
    use core::arch::x86_64::*;

    let shuffle: __m512i = _mm512_set_epi8(
        -1, 11, 10, 9, //
        -1, 8, 7, 6, //
        -1, 5, 4, 3, //
        -1, 2, 1, 0, //
        -1, 15, 14, 13, //
        -1, 12, 11, 10, //
        -1, 9, 8, 7, //
        -1, 6, 5, 4, //
        -1, 11, 10, 9, //
        -1, 8, 7, 6, //
        -1, 5, 4, 3, //
        -1, 2, 1, 0, //
        -1, 15, 14, 13, //
        -1, 12, 11, 10, //
        -1, 9, 8, 7, //
        -1, 6, 5, 4, //
    );

    let mut r = reg_dense.sub(4);
    let mut t = reg_raw;

    for _ in 0..HLL_REGISTERS / 64 {
        let x0 = _mm256_loadu_si256(r.cast());
        let x1 = _mm256_loadu_si256(r.add(24).cast());
        let x = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);
        let x = _mm512_shuffle_epi8(x, shuffle);

        let a1 = _mm512_and_si512(x, _mm512_set1_epi32(0x0000_003f));
        let a2 = _mm512_and_si512(x, _mm512_set1_epi32(0x0000_0fc0));
        let a3 = _mm512_and_si512(x, _mm512_set1_epi32(0x0003_f000));
        let a4 = _mm512_and_si512(x, _mm512_set1_epi32(0x00fc_0000));

        let a2 = _mm512_slli_epi32(a2, 2);
        let a3 = _mm512_slli_epi32(a3, 4);
        let a4 = _mm512_slli_epi32(a4, 6);

        let y1 = _mm512_or_si512(a1, a2);
        let y2 = _mm512_or_si512(a3, a4);
        let y = _mm512_or_si512(y1, y2);

        let z = _mm512_loadu_si512(t.cast());
        let z = _mm512_max_epu8(z, y);
        _mm512_storeu_si512(t.cast(), z);

        r = r.add(48);
        t = t.add(64);
    }
}

#[inline(always)]
unsafe fn compress(reg_dense: *mut u8, reg_raw: *const u8) {
    if const { HLL_BITS == 6 && HLL_REGISTERS % 64 == 0 }
        && is_simd_enabled()
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
    {
        return compress_avx512(reg_dense, reg_raw);
    }
    compress_scalar(reg_dense, reg_raw);
}

#[allow(clippy::cast_possible_truncation)]
unsafe fn compress_scalar(reg_dense: *mut u8, reg_raw: *const u8) {
    for i in 0..HLL_REGISTERS {
        let val = *reg_raw.add(i);
        set_register(reg_dense, i as u32, val);
    }
}

#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512bw")]
unsafe fn compress_avx512(reg_dense: *mut u8, reg_raw: *const u8) {
    use core::arch::x86_64::*;

    let indices = _mm512_setr_epi32(
        0, 3, 6, 9, 12, 15, 18, 21, //
        24, 27, 30, 33, 36, 39, 42, 45, //
    );

    let mut r = reg_raw;
    let mut t = reg_dense;

    for _ in 0..HLL_REGISTERS / 64 {
        let x = _mm512_loadu_si512(r.cast());

        let a1 = _mm512_and_si512(x, _mm512_set1_epi32(0x0000_003f));
        let a2 = _mm512_and_si512(x, _mm512_set1_epi32(0x0000_3f00));
        let a3 = _mm512_and_si512(x, _mm512_set1_epi32(0x003f_0000));
        let a4 = _mm512_and_si512(x, _mm512_set1_epi32(0x3f00_0000));

        let a2 = _mm512_srli_epi32(a2, 2);
        let a3 = _mm512_srli_epi32(a3, 4);
        let a4 = _mm512_srli_epi32(a4, 6);

        let y1 = _mm512_or_si512(a1, a2);
        let y2 = _mm512_or_si512(a3, a4);
        let y = _mm512_or_si512(y1, y2);

        _mm512_i32scatter_epi32(t.cast(), indices, y, 1);

        r = r.add(64);
        t = t.add(48);
    }
}
