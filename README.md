# redis-hyperloglog

Accelerate Redis PFMERGE command by 10 times.

```
======================================================================================================
Type             Ops/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
------------------------------------------------------------------------------------------------------
PFMERGE-scalar    9635.74        20.82536        20.09500        37.63100        40.19100       602.23
PFMERGE-avx2    122356.50         1.63272         1.57500         3.18300         5.15100      7647.28
```

This repository contains our experiment code, including the SIMD algorithms and benchmarks.

The C++ code tests the performance of different SIMD algorithms for optimizing the basic operations in HyperLogLog. The Rust code implements the dense encoding part of Redis HyperLogLog and applies SIMD optimization.

We ported the optimized dense encoding algorithm to Redis and tested it with memtier_benchmark. The results show that the SIMD optimization can achieve a 10x speedup compared to the scalar version.

Our fork:
+ https://github.com/Nugine/redis/tree/hll-simd
