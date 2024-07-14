use crate::HyperLogLog;

#[allow(clippy::cast_precision_loss)]
#[test]
fn merge3() {
    let cases: &[u64] = if cfg!(miri) {
        &[10] //
    } else {
        &[10, 100, 1000, 10000] //
    };

    for &n in cases {
        let mut hll1 = HyperLogLog::new();
        let mut hll2 = HyperLogLog::new();
        let mut hll3 = HyperLogLog::new();

        for i in 1..=n {
            hll1.insert(i.to_string().as_bytes());
            hll2.insert((i + n).to_string().as_bytes());
            hll3.insert((i + 2 * n).to_string().as_bytes());
        }

        let mut hll_merged = HyperLogLog::new();
        hll_merged.merge(&[hll1, hll2, hll3]);

        let count = hll_merged.count();
        let truth = n * 3;

        let err = (count as f64 - truth as f64) / (truth as f64);
        let err = err * 100.;

        println!("count: {count:>6}, truth: {truth:>6}, err: {err:.6}%");
        assert!(err.abs() < 0.67);
    }
}
