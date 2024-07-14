use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

pub const HLL_P: usize = 14;
pub const HLL_Q: usize = 64 - HLL_P;

pub const HLL_BITS: usize = 6; // ceil(log2(HLL_Q))

pub const HLL_REGISTERS: usize = 1 << HLL_P;

pub const HLL_HIST_LEN: usize = 1 << HLL_BITS;

#[allow(clippy::excessive_precision)]
pub const HLL_ALPHA_INF: f64 = 0.721_347_520_444_481_703_680;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HllRepr {
    Dense = 0,
}

#[allow(clippy::cast_possible_truncation)]
pub fn hll_pattern(hash: u64) -> (u32, u8) {
    const HLL_P_MASK: u32 = (1 << HLL_P) - 1;
    let index = (hash as u32) & HLL_P_MASK;

    let hash = (hash >> HLL_P) | (1 << HLL_Q);
    let count = hash.trailing_zeros().wrapping_add(1) as u8;

    (index, count)
}

#[allow(clippy::float_cmp)] // Redis uses strict cmp
pub fn hll_tau(mut x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    let mut z_prime: f64;
    let mut y = 1.0;
    let mut z = 1.0 - x;
    loop {
        x = x.sqrt();
        z_prime = z;
        y *= 0.5;
        z -= (1.0 - x).powi(2) * y;
        if z_prime != z {
            break;
        }
    }
    z / 3.0
}

#[allow(clippy::float_cmp)] // Redis uses strict cmp
pub fn hll_sigma(mut x: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    let mut z_prime: f64;
    let mut y = 1.0;
    let mut z = x;
    loop {
        x *= x;
        z_prime = z;
        z += x * y;
        y += y;
        if z_prime == z {
            break;
        }
    }
    z
}

static SIMD: AtomicBool = AtomicBool::new(true);

pub fn set_simd(enabled: bool) {
    SIMD.store(enabled, Ordering::SeqCst);
}

pub fn is_simd_enabled() -> bool {
    SIMD.load(Ordering::SeqCst)
}
