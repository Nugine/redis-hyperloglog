# redis-hyperloglog

Accelerate Redis PFCOUNT and PFMERGE by 12 times.

```
======================================================================================================
Type             Ops/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec 
------------------------------------------------------------------------------------------------------
PFCOUNT-scalar    5280.76        37.93893        34.81500        69.63100        73.72700       283.63
PFCOUNT-avx2     69802.99         2.85844         2.75100         5.53500         6.97500      3749.18
------------------------------------------------------------------------------------------------------
PFMERGE-scalar    9445.56        21.17554        20.09500        38.91100        41.21500       590.35
PFMERGE-avx2    120642.02         1.65367         1.59100         3.21500         5.11900      7540.13
------------------------------------------------------------------------------------------------------

CPU:    13th Gen Intel® Core™ i9-13900H × 20
Memory: 32.0 GiB
OS:     Ubuntu 22.04.5 LTS
```

This repository contains our experiment code, including the SIMD algorithms and benchmarks.

The C++ code tests the performance of different SIMD algorithms for optimizing the basic operations in HyperLogLog. The Rust code implements the dense encoding part of Redis HyperLogLog and applies SIMD optimization.

We ported the optimized dense encoding algorithm to Redis and [tested it with memtier_benchmark](./scripts/memtier.sh). The results show that the SIMD optimization can achieve a 12x speedup compared to the scalar version.

Our fork:
+ https://github.com/Nugine/redis/tree/hll-simd
