#![feature(unchecked_shifts)]
#![feature(stdarch_x86_avx512)]
#![feature(avx512_target_feature)]
#![deny(clippy::all, clippy::pedantic)]
#![allow(
    clippy::inline_always,
    clippy::module_name_repetitions,
    clippy::wildcard_imports,
    clippy::missing_panics_doc
)]

mod array;
mod config;
mod dense;
mod hash;

pub use self::config::{is_simd_enabled, set_simd};

use self::config::HllRepr;
use self::dense::HllDense;
use self::hash::murmurhash64a;

#[repr(transparent)]
pub struct HyperLogLog {
    ptr: *mut (),
}

impl HyperLogLog {
    #[must_use]
    pub fn new() -> Self {
        let ptr = HllDense::create().cast();
        Self { ptr }
    }

    pub fn clear(&mut self) {
        match self.repr() {
            HllRepr::Dense => unsafe { HllDense::clear(&mut *self.ptr.cast()) },
        }
    }

    pub fn insert(&mut self, key: &[u8]) -> bool {
        const SEED: u64 = 0xadc8_3b19;
        let hash = murmurhash64a(key, SEED);
        match self.repr() {
            HllRepr::Dense => unsafe { HllDense::insert(&mut *self.ptr.cast(), hash) },
        }
    }

    pub fn count(&mut self) -> u64 {
        match self.repr() {
            HllRepr::Dense => unsafe { HllDense::count(&mut *self.ptr.cast()) },
        }
    }

    pub fn merge(&mut self, sources: &[&Self]) {
        assert!(self.repr() == HllRepr::Dense);
        for src in sources {
            assert!(src.repr() == HllRepr::Dense);
        }
        let sources: &[&&HllDense] = unsafe { slice_cast(sources) };
        unsafe { HllDense::merge(&mut *self.ptr.cast(), sources) }
    }
}

impl HyperLogLog {
    fn repr(&self) -> HllRepr {
        unsafe { self.ptr.cast::<HllRepr>().read() }
    }
}

impl Drop for HyperLogLog {
    fn drop(&mut self) {
        match self.repr() {
            HllRepr::Dense => unsafe { HllDense::destroy(self.ptr.cast()) },
        }
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}

unsafe fn slice_cast<T, U>(slice: &[T]) -> &[U] {
    let len = slice.len();
    let ptr = slice.as_ptr().cast();
    std::slice::from_raw_parts(ptr, len)
}
