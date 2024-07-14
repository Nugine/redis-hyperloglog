#!/bin/bash -ex
export RUSTFLAGS="-C target-cpu=native"
cargo criterion
