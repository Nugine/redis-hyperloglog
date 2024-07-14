#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::inline_always, clippy::module_name_repetitions, clippy::wildcard_imports)]

mod array;
mod config;
mod dense;
mod hash;

use self::config::HllRepr;
use self::dense::HllDense;

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
        todo!()
    }

    pub fn count(&mut self) -> u64 {
        todo!()
    }

    pub fn merge(&mut self, sources: &[&Self]) {
        todo!()
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
