use std::alloc::alloc_zeroed;
use std::alloc::dealloc;
use std::alloc::handle_alloc_error;
use std::alloc::Layout;
use std::ptr;

use crate::array::UnsafeArray;
use crate::config::*;

const DENSE_PAD_LEN: usize = 16;
const DENSE_REGISTERS_LEN: usize = (HLL_REGISTERS * HLL_BITS + 7) / 8 + DENSE_PAD_LEN;

#[repr(C)]
pub struct HllDense {
    repr: HllRepr,
    cmin: u8,
    _pad: [u8; 6],
    card: u64,
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
        (*this).card = 0;
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
}
