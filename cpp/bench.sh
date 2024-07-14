#!/bin/bash -ex
g++ cpp/bench.cpp -O3 -march=native -Wall -Wextra -std=c++20 -o logs/a.out
./logs/a.out | tee logs/cpp_bench.log
