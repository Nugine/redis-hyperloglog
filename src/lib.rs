#![deny(clippy::all, clippy::pedantic)]
#![allow(clippy::inline_always, clippy::module_name_repetitions)]

mod array;
mod config;
mod hash;

use self::config::HllEncoding;

#[repr(transparent)]
pub struct HyperLogLog {
    ptr: *mut (),
}

impl HyperLogLog {
    #[must_use]
    pub fn new() -> Self {
        todo!()
    }

    pub fn clear(&mut self) {
        todo!()
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
    fn encoding(&self) -> HllEncoding {
        unsafe { self.ptr.cast::<HllEncoding>().read() }
    }
}

impl Drop for HyperLogLog {
    fn drop(&mut self) {
        todo!()
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}
