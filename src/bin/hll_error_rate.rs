use redis_hyperloglog::HyperLogLog;

use std::collections::HashSet;
use std::fs;
use std::io;

use clap::Parser;
use indicatif::ProgressStyle;
use ndarray::Array;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

#[derive(clap::Parser, Debug)]
struct Args {
    #[clap(short, default_value = "100000000")]
    n: usize,

    #[clap(short, long, default_value = "16")]
    rounds: usize,

    #[clap(short, long, default_value = "16")]
    batch_size: usize,

    #[clap(long, default_value = "results.json")]
    save: String,
}

#[allow(clippy::needless_range_loop)]
fn gen_rand_seq(n: usize) -> Vec<u64> {
    let mut set = HashSet::with_capacity(n);
    let mut vals = vec![0u64; n];
    for i in 0..n {
        let mut val = rand::random::<u64>();
        while !set.insert(val) {
            val = rand::random::<u64>();
        }
        vals[i] = val;
    }
    vals
}

#[allow(clippy::cast_precision_loss, clippy::cast_lossless, clippy::needless_range_loop)]
fn run_error_rate(n: usize) -> (f64, f64, f64) {
    let vals = gen_rand_seq(n);

    let mut hll = HyperLogLog::new();
    assert_eq!(hll.count(), 0);

    let mut max_err_abs: f64 = 0.0;
    let mut sum_err_abs: f64 = 0.0;

    for i in 1..=n {
        hll.insert(&vals[i - 1].to_be_bytes());
        let count = hll.count();

        let err = (count as f64 - f64::from(i as u32)) / f64::from(i as u32);

        max_err_abs = max_err_abs.max(err.abs());
        sum_err_abs += err.abs();
    }

    let last_err = (hll.count() as f64 - n as f64) / n as f64;

    let avg_err = sum_err_abs / (n as f64);
    (max_err_abs, avg_err, last_err)
}

#[allow(clippy::needless_range_loop)]
fn main() -> io::Result<()> {
    let args = Args::parse();
    let Args {
        n,
        rounds,
        batch_size,
        ref save,
    } = args;
    println!("{args:?}");

    let mut total_results = Vec::with_capacity(rounds);

    let pbar = indicatif::ProgressBar::new(rounds as u64);
    pbar.set_style(
        ProgressStyle::with_template("[{elapsed_precise}][{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap(),
    );

    let mut r = 0;
    while r < rounds {
        let batch = (rounds - r).min(batch_size);

        let results = (0..batch).into_par_iter().map(|_| run_error_rate(n)).collect::<Vec<_>>();

        for i in 0..batch {
            let round = r + i + 1;
            let (max_err, avg_err, last_err) = results[i];
            pbar.println(format!(
                "round: {:>2}, max_err: {:.4}%, avg_err: {:.4}%, last_err: {:>6.4}%",
                round,
                max_err * 100.0,
                avg_err * 100.0,
                last_err * 100.0
            ));
        }

        total_results.extend(results);

        r += batch;
        pbar.inc(batch as u64);
    }

    {
        let mut file = fs::File::create(save)?;
        serde_json::to_writer(&mut file, &total_results)?;
    }

    {
        let last_errors = Array::from_iter(total_results.iter().map(|&(_, _, last_err)| last_err));
        let sample_var = last_errors.var(1.0);
        println!("sample variance: {sample_var:.4}");
    }

    println!("done");

    Ok(())
}
