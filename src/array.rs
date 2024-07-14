use std::fmt;
use std::fmt::Debug;
use std::ops::Index;
use std::ops::IndexMut;

const _: () = assert!(cfg!(target_pointer_width = "64"));

/// It is unsafe to create an instance of this struct,
/// because it allows reading and writing without bounds checks.
#[repr(transparent)]
pub struct UnsafeArray<T, const N: usize> {
    data: [T; N],
}

impl<T: Copy, const N: usize> UnsafeArray<T, N> {
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

impl<T: Debug, const N: usize> Debug for UnsafeArray<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <[T; N] as Debug>::fmt(&self.data, f)
    }
}

impl<T, const N: usize> Index<usize> for UnsafeArray<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < N);
        unsafe { self.data.get_unchecked(index) }
    }
}

impl<T, const N: usize> IndexMut<usize> for UnsafeArray<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < N);
        unsafe { self.data.get_unchecked_mut(index) }
    }
}

impl<T, const N: usize> Index<u32> for UnsafeArray<T, N> {
    type Output = T;

    fn index(&self, index: u32) -> &Self::Output {
        debug_assert!((index as usize) < N);
        unsafe { self.data.get_unchecked(index as usize) }
    }
}

impl<T, const N: usize> IndexMut<u32> for UnsafeArray<T, N> {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        debug_assert!((index as usize) < N);
        unsafe { self.data.get_unchecked_mut(index as usize) }
    }
}

impl<T, const N: usize> Index<u8> for UnsafeArray<T, N> {
    type Output = T;

    fn index(&self, index: u8) -> &Self::Output {
        debug_assert!((index as usize) < N);
        unsafe { self.data.get_unchecked(index as usize) }
    }
}

impl<T, const N: usize> IndexMut<u8> for UnsafeArray<T, N> {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        debug_assert!((index as usize) < N);
        unsafe { self.data.get_unchecked_mut(index as usize) }
    }
}
