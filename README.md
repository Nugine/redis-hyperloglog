# redis-hyperloglog

Accelerate Redis PFMERGE command by 10 times.

```
======================================================================================================
Type             Ops/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
------------------------------------------------------------------------------------------------------
PFMERGE-scalar    5100.65        39.35689        35.58300        74.23900        90.11100       318.79 
PFMERGE-avx2     69189.82         2.97691         2.83100         5.82300        10.75100      4324.36
```

This repository contains our experiment code, including the SIMD algorithms and benchmarks.

The C++ code tests the performance of different SIMD algorithms for optimizing the basic operations in HyperLogLog. The Rust code implements the dense encoding part of Redis HyperLogLog and applies SIMD optimization.

We ported the optimized dense encoding algorithm to Redis and tested it with memtier_benchmark. The results show that the SIMD optimization can achieve a 10x speedup compared to the scalar version.

Our fork:
+ https://github.com/Nugine/redis/tree/avx2
+ https://github.com/Nugine/redis/tree/avx512
