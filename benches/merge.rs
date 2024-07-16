use redis_hyperloglog::HyperLogLog;

use criterion::{black_box, criterion_group, criterion_main, AxisScale, PlotConfiguration};
use criterion::{BenchmarkId, Criterion};

pub fn bench_merge(c: &mut Criterion) {
    dbg!(is_x86_feature_detected!("avx2"));
    dbg!(is_x86_feature_detected!("avx512f"));
    dbg!(is_x86_feature_detected!("avx512bw"));

    let mut group = c.benchmark_group("merge");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let nums = [2, 3, 7, 30, 60, 90];

    for n in nums {
        let mut hlls = Vec::new();
        for i in 0u32..n {
            let mut hll = HyperLogLog::new();
            hll.insert(&i.to_be_bytes());
            hlls.push(hll);
        }

        let mut dst = HyperLogLog::new();
        dst.insert(&n.to_be_bytes());

        redis_hyperloglog::set_simd(true);
        group.bench_with_input(BenchmarkId::new("merge-simd", n), &n, |b, _| {
            b.iter(|| dst.merge(black_box(hlls.as_slice())));
        });

        redis_hyperloglog::set_simd(false);
        group.bench_with_input(BenchmarkId::new("merge-scalar", n), &n, |b, _| {
            b.iter(|| dst.merge(black_box(hlls.as_slice())));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_merge);
criterion_main!(benches);
