#!/bin/bash -ex
export RUSTFLAGS="-C target-cpu=native"
cargo fetch
cargo build --benches --release
cargo criterion
