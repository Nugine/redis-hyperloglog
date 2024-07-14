#[allow(clippy::cast_ptr_alignment, clippy::cast_possible_truncation)]
pub fn murmurhash64a(key: &[u8], seed: u64) -> u64 {
    let len = key.len();
    let key = key.as_ptr();
    unsafe {
        let m: u64 = 0xc6a4_a793_5bd1_e995;
        let r: u32 = 47;

        let mut h: u64 = seed ^ ((len as u64).wrapping_mul(m));

        let mut data = key.cast::<u64>();
        let end = data.add(len / 8);

        while data != end {
            let mut k = data.read_unaligned();
            data = data.add(1);

            k = k.to_le();

            k = k.wrapping_mul(m);
            k ^= k >> r;
            k = k.wrapping_mul(m);

            h ^= k;
            h = h.wrapping_mul(m);
        }

        let data = data.cast::<u8>();
        let rem = (len & 7) as u32;
        if rem == 7 {
            h ^= (u64::from(*data.add(6))) << 48;
        }
        if rem >= 6 {
            h ^= (u64::from(*data.add(5))) << 40;
        }
        if rem >= 5 {
            h ^= (u64::from(*data.add(4))) << 32;
        }
        if rem >= 4 {
            h ^= (u64::from(*data.add(3))) << 24;
        }
        if rem >= 3 {
            h ^= (u64::from(*data.add(2))) << 16;
        }
        if rem >= 2 {
            h ^= (u64::from(*data.add(1))) << 8;
        }
        if rem >= 1 {
            h ^= u64::from(*data);
            h = h.wrapping_mul(m);
        }

        h ^= h >> r;
        h = h.wrapping_mul(m);
        h ^= h >> r;

        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_value() {
        let seed = 0xadc8_3b19;

        let cases: &[(&str, u64)] = &[
            ("7", 5_554_161_992_923_675_127),
            ("21", 12_846_450_894_857_633_433),
            ("1411", 3_845_932_236_355_773_924),
        ];

        for &(key, expected) in cases {
            let hash = murmurhash64a(key.as_ref(), seed);
            assert_eq!(hash, expected, "key: {key}");
        }
    }
}
