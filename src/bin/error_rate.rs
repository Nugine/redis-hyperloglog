use redis_hyperloglog::HyperLogLog;

use rand::seq::SliceRandom;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

#[allow(clippy::cast_precision_loss, clippy::cast_lossless, clippy::needless_range_loop)]
fn run_error_rate(n: usize) -> (f64, f64) {
    let mut vals = vec![0u64; n + 1];
    for i in 1..=n {
        vals[i] = i as u64;
    }
    vals.as_mut_slice().shuffle(&mut rand::thread_rng());

    let mut hll = HyperLogLog::new();
    assert_eq!(hll.count(), 0);

    let mut max_err_abs: f64 = 0.0;
    let mut sum_err_abs: f64 = 0.0;

    for i in 1..=n {
        hll.insert(&vals[i].to_be_bytes());
        let count = hll.count();

        let err = (count as f64 - f64::from(i as u32)) / f64::from(i as u32);
        let err = err * 100.;

        max_err_abs = max_err_abs.max(err.abs());
        sum_err_abs += err.abs();
    }

    let avg_err = sum_err_abs / (n as f64);
    (max_err_abs, avg_err)
}

fn main() {
    let n: usize = 1_0000_0000;
    let rounds: usize = 16;

    let results = (1..=rounds).into_par_iter().map(|_| run_error_rate(n)).collect::<Vec<_>>();

    println!("n: {n}");
    for round in 1..=rounds {
        let (max_err, avg_err) = results[round - 1];
        println!("round: {round:>2}, max_err: {max_err:.4}%, avg_err: {avg_err:.4}%");
    }
}
