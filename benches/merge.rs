use redis_hyperloglog::HyperLogLog;

use criterion::{black_box, criterion_group, criterion_main};
use criterion::{BenchmarkId, Criterion};

pub fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge");

    let nums = [2, 3, 7, 30, 60, 90];

    for n in nums {
        let mut hlls = Vec::new();
        for i in 0u32..n {
            let mut hll = HyperLogLog::new();
            hll.insert(&i.to_be_bytes());
            hlls.push(hll);
        }

        let mut dst = HyperLogLog::new();

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
