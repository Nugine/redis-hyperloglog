#!/bin/bash -ex
CXX='g++ -O3 -march=native -Wall -Wextra -std=c++20'
if lscpu | grep -q avx512; then
    $CXX cpp/bench.cpp -o logs/a.out
else
    $CXX cpp/bench.cpp -o logs/a.out -DNO_AVX512
fi
./logs/a.out | tee logs/cpp_bench.log
