#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <random>

#include <immintrin.h>

#define HLL_P 14 /* The greater is P, the smaller the error. */
#define HLL_Q                                                                  \
    (64 - HLL_P) /* The number of bits of the hash value used for              \
                    determining the number of leading zeros. */
#define HLL_REGISTERS (1 << HLL_P)     /* With P=14, 16384 registers. */
#define HLL_P_MASK (HLL_REGISTERS - 1) /* Mask to index register. */
#define HLL_BITS 6 /* Enough to count up to 63 leading zeroes. */
#define HLL_REGISTER_MAX ((1 << HLL_BITS) - 1)
#define HLL_HDR_SIZE sizeof(struct hllhdr)
#define HLL_DENSE_SIZE (HLL_HDR_SIZE + ((HLL_REGISTERS * HLL_BITS + 7) / 8))
#define HLL_DENSE 0  /* Dense encoding. */
#define HLL_SPARSE 1 /* Sparse encoding. */
#define HLL_RAW 255  /* Only used internally, never exposed. */
#define HLL_MAX_ENCODING 1

#define HLL_DENSE_GET_REGISTER(target, p, regnum)                              \
    do {                                                                       \
        uint8_t *_p = (uint8_t *)p;                                            \
        unsigned long _byte = regnum * HLL_BITS / 8;                           \
        unsigned long _fb = regnum * HLL_BITS & 7;                             \
        unsigned long _fb8 = 8 - _fb;                                          \
        unsigned long b0 = _p[_byte];                                          \
        unsigned long b1 = _p[_byte + 1];                                      \
        target = ((b0 >> _fb) | (b1 << _fb8)) & HLL_REGISTER_MAX;              \
    } while (0)

/* Set the value of the register at position 'regnum' to 'val'.
 * 'p' is an array of unsigned bytes. */
#define HLL_DENSE_SET_REGISTER(p, regnum, val)                                 \
    do {                                                                       \
        uint8_t *_p = (uint8_t *)p;                                            \
        unsigned long _byte = (regnum) * HLL_BITS / 8;                         \
        unsigned long _fb = (regnum) * HLL_BITS & 7;                           \
        unsigned long _fb8 = 8 - _fb;                                          \
        unsigned long _v = (val);                                              \
        _p[_byte] &= ~(HLL_REGISTER_MAX << _fb);                               \
        _p[_byte] |= _v << _fb;                                                \
        _p[_byte + 1] &= ~(HLL_REGISTER_MAX >> _fb8);                          \
        _p[_byte + 1] |= _v >> _fb8;                                           \
    } while (0)

void merge_base(uint8_t *reg_raw, const uint8_t *reg_dense) {
    uint8_t val;
    for (int i = 0; i < HLL_REGISTERS; i++) {
        HLL_DENSE_GET_REGISTER(val, reg_dense, i);
        if (val > reg_raw[i]) {
            reg_raw[i] = val;
        }
    }
}

void merge_avx512_1(uint8_t *reg_raw, const uint8_t *reg_dense) {
    const __m512i shuffle = _mm512_set_epi8( //
        0x80, 11, 10, 9,                     //
        0x80, 8, 7, 6,                       //
        0x80, 5, 4, 3,                       //
        0x80, 2, 1, 0,                       //
        0x80, 15, 14, 13,                    //
        0x80, 12, 11, 10,                    //
        0x80, 9, 8, 7,                       //
        0x80, 6, 5, 4,                       //
        0x80, 11, 10, 9,                     //
        0x80, 8, 7, 6,                       //
        0x80, 5, 4, 3,                       //
        0x80, 2, 1, 0,                       //
        0x80, 15, 14, 13,                    //
        0x80, 12, 11, 10,                    //
        0x80, 9, 8, 7,                       //
        0x80, 6, 5, 4                        //
    );

    const uint8_t *r = reg_dense - 4;
    const uint8_t *t = reg_raw;

    for (int i = 0; i < HLL_REGISTERS / 64; ++i) {
        __m256i x0, x1;
        __m512i x;
        x0 = _mm256_loadu_si256((__m256i *)r);
        x1 = _mm256_loadu_si256((__m256i *)(r + 24));

        x = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);
        x = _mm512_shuffle_epi8(x, shuffle);

        __m512i a1, a2, a3, a4;
        a1 = _mm512_and_si512(x, _mm512_set1_epi32(0x0000003f));
        a2 = _mm512_and_si512(x, _mm512_set1_epi32(0x00000fc0));
        a3 = _mm512_and_si512(x, _mm512_set1_epi32(0x0003f000));
        a4 = _mm512_and_si512(x, _mm512_set1_epi32(0x00fc0000));

        a2 = _mm512_slli_epi32(a2, 2);
        a3 = _mm512_slli_epi32(a3, 4);
        a4 = _mm512_slli_epi32(a4, 6);

        __m512i y1, y2, y;
        y1 = _mm512_or_si512(a1, a2);
        y2 = _mm512_or_si512(a3, a4);
        y = _mm512_or_si512(y1, y2);

        __m512i z = _mm512_loadu_si512((__m512i *)t);

        z = _mm512_max_epu8(z, y);

        _mm512_storeu_si512((__m512i *)t, z);

        r += 48;
        t += 64;
    }
}

void merge_avx512_2(uint8_t *reg_raw, const uint8_t *reg_dense) {
    const __m512i indices = _mm512_setr_epi32( //
        0, 3, 6, 9, 12, 15, 18, 21,            //
        24, 27, 30, 33, 36, 39, 42, 45         //
    );

    const uint8_t *r = reg_dense;
    const uint8_t *t = reg_raw;

    for (int i = 0; i < HLL_REGISTERS / 64; ++i) {
        __m512i x = _mm512_i32gather_epi32(indices, r, 1);

        __m512i a1, a2, a3, a4;
        a1 = _mm512_and_si512(x, _mm512_set1_epi32(0x0000003f));
        a2 = _mm512_and_si512(x, _mm512_set1_epi32(0x00000fc0));
        a3 = _mm512_and_si512(x, _mm512_set1_epi32(0x0003f000));
        a4 = _mm512_and_si512(x, _mm512_set1_epi32(0x00fc0000));

        a2 = _mm512_slli_epi32(a2, 2);
        a3 = _mm512_slli_epi32(a3, 4);
        a4 = _mm512_slli_epi32(a4, 6);

        __m512i y1, y2, y;
        y1 = _mm512_or_si512(a1, a2);
        y2 = _mm512_or_si512(a3, a4);
        y = _mm512_or_si512(y1, y2);

        __m512i z = _mm512_loadu_si512((__m512i *)t);

        z = _mm512_max_epu8(z, y);

        _mm512_storeu_si512((__m512i *)t, z);

        r += 48;
        t += 64;
    }
}

void compress_base(uint8_t *reg_dense, const uint8_t *reg_raw) {
    for (int i = 0; i < HLL_REGISTERS; i++) {
        HLL_DENSE_SET_REGISTER(reg_dense, i, reg_raw[i]);
    }
}

void compress_avx512(uint8_t *reg_dense, const uint8_t *reg_raw) {
    const __m512i indices = _mm512_setr_epi32( //
        0, 3, 6, 9, 12, 15, 18, 21,            //
        24, 27, 30, 33, 36, 39, 42, 45         //
    );

    const uint8_t *r = reg_raw;
    uint8_t *t = reg_dense;

    for (int i = 0; i < HLL_REGISTERS / 64; ++i) {
        __m512i x = _mm512_loadu_si512((__m512i *)r);

        __m512i a1, a2, a3, a4;
        a1 = _mm512_and_si512(x, _mm512_set1_epi32(0x0000003f));
        a2 = _mm512_and_si512(x, _mm512_set1_epi32(0x00003f00));
        a3 = _mm512_and_si512(x, _mm512_set1_epi32(0x003f0000));
        a4 = _mm512_and_si512(x, _mm512_set1_epi32(0x3f000000));

        a2 = _mm512_srli_epi32(a2, 2);
        a3 = _mm512_srli_epi32(a3, 4);
        a4 = _mm512_srli_epi32(a4, 6);

        __m512i y1, y2, y;
        y1 = _mm512_or_si512(a1, a2);
        y2 = _mm512_or_si512(a3, a4);
        y = _mm512_or_si512(y1, y2);

        _mm512_i32scatter_epi32((__m512i *)t, indices, y, 1);

        r += 64;
        t += 48;
    }
}

#define HLL_DENSE_REG_LEN (HLL_REGISTERS * HLL_BITS / 8)

void histogram_base_1(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense;

    unsigned int r0, r1, r2, r3;
    for (int j = 0; j < 4096; ++j) {
        r0 = r[0] & 63;
        r1 = (r[0] >> 6 | r[1] << 2) & 63;
        r2 = (r[1] >> 4 | r[2] << 4) & 63;
        r3 = (r[2] >> 2) & 63;

        hist[r0]++;
        hist[r1]++;
        hist[r2]++;
        hist[r3]++;

        r += 3;
    }
}

void histogram_base_2(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense, *end = reg_dense + HLL_DENSE_REG_LEN;

    unsigned int r0, r1, r2, r3;
    for (; r < end; r += 3) {
        r0 = r[0] & 63;
        r1 = (r[0] >> 6 | r[1] << 2) & 63;
        r2 = (r[1] >> 4 | r[2] << 4) & 63;
        r3 = (r[2] >> 2) & 63;

        hist[r0]++;
        hist[r1]++;
        hist[r2]++;
        hist[r3]++;
    }
}

void histogram_unroll(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense;

    unsigned int r0, r1, r2, r3, r4, r5, r6, r7;
    unsigned int r8, r9, r10, r11, r12, r13, r14, r15;
    for (int j = 0; j < 1024; ++j) {
        r0 = r[0] & 63;
        r1 = (r[0] >> 6 | r[1] << 2) & 63;
        r2 = (r[1] >> 4 | r[2] << 4) & 63;
        r3 = (r[2] >> 2) & 63;
        r4 = r[3] & 63;
        r5 = (r[3] >> 6 | r[4] << 2) & 63;
        r6 = (r[4] >> 4 | r[5] << 4) & 63;
        r7 = (r[5] >> 2) & 63;
        r8 = r[6] & 63;
        r9 = (r[6] >> 6 | r[7] << 2) & 63;
        r10 = (r[7] >> 4 | r[8] << 4) & 63;
        r11 = (r[8] >> 2) & 63;
        r12 = r[9] & 63;
        r13 = (r[9] >> 6 | r[10] << 2) & 63;
        r14 = (r[10] >> 4 | r[11] << 4) & 63;
        r15 = (r[11] >> 2) & 63;

        hist[r0]++;
        hist[r1]++;
        hist[r2]++;
        hist[r3]++;
        hist[r4]++;
        hist[r5]++;
        hist[r6]++;
        hist[r7]++;
        hist[r8]++;
        hist[r9]++;
        hist[r10]++;
        hist[r11]++;
        hist[r12]++;
        hist[r13]++;
        hist[r14]++;
        hist[r15]++;

        r += 12;
    }
}

/**
load
{????|AAAB|BBCC|CDDD|EEEF|FFGG|GHHH|????}

shuffle
{bbaaaaaa|ccccbbbb|ddddddcc|00000000} x8

and -> 00aaaaaa
and, slli -> 00bbbbbb
and, slli -> 00cccccc
and, slli -> 00dddddd
or,or,or -> {00aaaaaa|00bbbbbb|00cccccc|00dddddd} x8
 */
const __m256i avx2_shuffle = _mm256_setr_epi8( //
    4, 5, 6, 0x80,                             //
    7, 8, 9, 0x80,                             //
    10, 11, 12, 0x80,                          //
    13, 14, 15, 0x80,                          //
    0, 1, 2, 0x80,                             //
    3, 4, 5, 0x80,                             //
    6, 7, 8, 0x80,                             //
    9, 10, 11, 0x80                            //
);
void histogram_avx2_1(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense - 4;
    for (int j = 0; j < 512; ++j) {
        __m256i x0 = _mm256_loadu_si256((__m256i *)r);
        __m256i x1 = _mm256_shuffle_epi8(x0, avx2_shuffle);

#ifdef DEBUG
        uint8_t dbg[32];
        _mm256_storeu_si256((__m256i *)dbg, x1);
        int k = 4;
        for (int i = 0; i < 8; ++i) {
            // fprintf(stderr, "i = %d, k = %d\n", i, k);
            assert(dbg[i * 4 + 0] == r[k++]);
            assert(dbg[i * 4 + 1] == r[k++]);
            assert(dbg[i * 4 + 2] == r[k++]);
            assert(dbg[i * 4 + 3] == 0);
        }
#endif

        __m256i a1, a2, a3, a4;
        a1 = _mm256_and_si256(x1, _mm256_set1_epi32(0x0000003f));
        a2 = _mm256_and_si256(x1, _mm256_set1_epi32(0x00000fc0));
        a3 = _mm256_and_si256(x1, _mm256_set1_epi32(0x0003f000));
        a4 = _mm256_and_si256(x1, _mm256_set1_epi32(0x00fc0000));

        a2 = _mm256_slli_epi32(a2, 2);
        a3 = _mm256_slli_epi32(a3, 4);
        a4 = _mm256_slli_epi32(a4, 6);

        __m256i y, y1, y2;
        y1 = _mm256_or_si256(a1, a2);
        y2 = _mm256_or_si256(a3, a4);
        y = _mm256_or_si256(y1, y2);

        uint8_t *t = (uint8_t *)(&y);

#ifdef DEBUG
        for (int i = 0; i < 32; ++i) {
            assert(t[i] < 64);
            uint8_t val;
            HLL_DENSE_GET_REGISTER(val, reg_dense, (j * 32 + i));
            assert(t[i] == val);
        }
#endif

        hist[t[0]]++, hist[t[1]]++, hist[t[2]]++, hist[t[3]]++;
        hist[t[4]]++, hist[t[5]]++, hist[t[6]]++, hist[t[7]]++;
        hist[t[8]]++, hist[t[9]]++, hist[t[10]]++, hist[t[11]]++;
        hist[t[12]]++, hist[t[13]]++, hist[t[14]]++, hist[t[15]]++;
        hist[t[16]]++, hist[t[17]]++, hist[t[18]]++, hist[t[19]]++;
        hist[t[20]]++, hist[t[21]]++, hist[t[22]]++, hist[t[23]]++;
        hist[t[24]]++, hist[t[25]]++, hist[t[26]]++, hist[t[27]]++;
        hist[t[28]]++, hist[t[29]]++, hist[t[30]]++, hist[t[31]]++;

        r += 24;
    }
}

class BenchmarkGroup {
  private:
    std::vector<std::function<void()>> functions;
    std::vector<std::string> names;
    std::vector<int> order;
    std::vector<double> runtime;

  public:
    BenchmarkGroup() {}

    void add(std::string name, std::function<void()> function) {
        names.push_back(std::move(name));
        functions.push_back(function);
        order.push_back(names.size() - 1);
    }

    void run(int rounds) {
        std::mt19937 rng(std::random_device{}());
        std::shuffle(order.begin(), order.end(), rng);

        int num = order.size();
        runtime.clear();
        runtime.resize(num);
        for (int i = 0; i < num; i++) {
            int idx = order[i];
            auto &function = functions[idx];
            const auto &name = names[idx];

            printf("%-20s: ", name.c_str());
            fflush(stdout);

            clock_t start = clock();
            for (int r = 0; r < rounds; ++r) {
                function();
            }
            clock_t end = clock();

            double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
            runtime[idx] = elapsed;

            printf("%.6fs\n", elapsed);
        }
    }

    void summary() {
        printf("---summary---\n");
        int num = order.size();
        for (int i = 0; i < num; i++) {
            printf("%-20s: %.6fs\n", names[i].c_str(), runtime[i]);
        }
    }
};

static uint8_t buf1[HLL_REGISTERS * 2];
static uint8_t buf2[HLL_REGISTERS * 2];
static uint8_t buf3[HLL_REGISTERS * 2];
static uint8_t buf4[HLL_REGISTERS * 2];

int check_merge(const uint8_t *lhs, const uint8_t *rhs) {
    for (int i = 0; i < HLL_REGISTERS; i++) {
        if (lhs[i] != rhs[i]) {
            return i;
        }
    }
    return -1;
}

void bench_merge(int rounds, int seed) {
    printf("------bench_merge------\n");

    srand(seed);

    uint8_t *reg_raw = buf1;
    uint8_t *reg_dense = buf2 + 64;

    printf("verify\n");
    for (int r = 0; r < rounds / 10; ++r) {
        for (int i = 0; i < HLL_REGISTERS; i++) {
            reg_raw[i] = rand();
        }
        for (int i = 0; i < HLL_REGISTERS; i++) {
            reg_dense[i] = rand();
        }

        memcpy(buf3, reg_raw, HLL_REGISTERS);
        merge_base(buf3, reg_dense);

        void (*funcs[2])(uint8_t *, const uint8_t *) = {
            merge_avx512_1, //
            merge_avx512_2, //
        };

        for (int j = 0; j < 2; ++j) {
            memcpy(buf4, reg_raw, HLL_REGISTERS);
            funcs[j](buf4, reg_dense);
            int idx = check_merge(buf3, buf4);
            if (idx >= 0) {
                uint8_t val1;
                uint8_t val2;
                val1 = reg_raw[idx];
                HLL_DENSE_GET_REGISTER(val2, reg_dense, idx);

                fprintf(stderr, "error: %d, %d, %d; %d %d\n", idx, buf3[idx],
                        buf4[idx], val1, val2);
                exit(1);
            }
        }
    }

    BenchmarkGroup group;
    group.add("merge_base", [=]() {
        memset(reg_raw, 0, HLL_REGISTERS);
        merge_base(reg_raw, reg_dense);
    });
    group.add("merge_avx512_1", [=]() {
        memset(reg_raw, 0, HLL_REGISTERS);
        merge_avx512_1(reg_raw, reg_dense);
    });
    group.add("merge_avx512_2", [=]() {
        memset(reg_raw, 0, HLL_REGISTERS);
        merge_avx512_2(reg_raw, reg_dense);
    });

    printf("benchmark\n");
    group.run(rounds);
    group.summary();

    printf("-----------------------\n");
}

int check_compress(const uint8_t *lhs, const uint8_t *rhs) {
    for (int i = 0; i < HLL_DENSE_REG_LEN; i++) {
        if (lhs[i] != rhs[i]) {
            return i;
        }
    }
    return -1;
}

void bench_compress(int rounds, int seed) {
    printf("------bench_compress------\n");

    srand(seed);

    uint8_t *reg_raw = buf1;
    uint8_t *reg_dense = buf2 + 64;

    printf("verify\n");
    for (int r = 0; r < rounds / 10; ++r) {
        for (int i = 0; i < HLL_REGISTERS; i++) {
            reg_raw[i] = rand();
        }

        compress_base(buf3, reg_raw);

        void (*funcs[1])(uint8_t *, const uint8_t *) = {
            compress_avx512, //
        };

        for (int j = 0; j < 1; ++j) {
            funcs[j](buf4, reg_raw);
            int idx = check_compress(buf3, buf4);
            if (idx >= 0) {
                fprintf(stderr, "error: %d, %d, %d\n", idx, buf3[idx],
                        buf4[idx]);
                exit(1);
            }
        }
    }

    BenchmarkGroup group;
    group.add("compress_base", [=]() {
        compress_base(reg_dense, reg_raw); //
    });
    group.add("compress_avx512", [=]() {
        compress_avx512(reg_dense, reg_raw); //
    });

    printf("benchmark\n");
    group.run(rounds);
    group.summary();

    printf("-----------------------\n");
}

int check_histogram(const int *lhs, const int *rhs) {
    for (int i = 0; i < 64; i++) {
        if (lhs[i] != rhs[i]) {
            return i;
        }
    }
    return -1;
}

static int hist1[64];
static int hist2[64];

void bench_histogram(int rounds, int seed) {
    printf("------bench_histogram------\n");

    srand(seed);

    uint8_t *reg_dense = buf1 + 64;

    printf("verify\n");
    for (int r = 0; r < rounds / 10; ++r) {
        for (int i = 0; i < HLL_DENSE_REG_LEN; i++) {
            reg_dense[i] = rand();
        }

        memset(hist1, 0, sizeof(hist1));
        histogram_base_1(reg_dense, hist1);

        void (*funcs[3])(const uint8_t *, int *) = {
            histogram_base_2, //
            histogram_unroll, //
            histogram_avx2_1, //
        };

        for (int j = 0; j < 3; ++j) {
            memset(hist2, 0, sizeof(hist2));
            funcs[j](reg_dense, hist2);
            int idx = check_histogram(hist1, hist2);
            if (idx >= 0) {
                fprintf(stderr, "error: %d, %d, %d, %d\n", j, idx, hist1[idx],
                        hist2[idx]);
                exit(1);
            }
        }
    }

    BenchmarkGroup group;
    group.add("histogram_base_1", [=]() {
        histogram_base_1(reg_dense, hist1); //
    });
    group.add("histogram_base_2", [=]() {
        histogram_base_2(reg_dense, hist1); //
    });
    group.add("histogram_unroll", [=]() {
        histogram_unroll(reg_dense, hist1); //
    });
    group.add("histogram_avx2_1", [=]() {
        histogram_avx2_1(reg_dense, hist1); //
    });

    printf("benchmark\n");
    group.run(rounds);
    group.summary();

    printf("-----------------------\n");
}

int main() {
#ifndef ROUNDS
    int rounds = 1e5;
#else
    int rounds = ROUNDS;
#endif

#ifndef SEED
    int seed = time(NULL);
#else
    int seed = SEED;
#endif

    printf("rounds: %d\n", rounds);
    printf("seed: %d\n", seed);

    bench_histogram(rounds, seed);
    bench_merge(rounds, seed);
    bench_compress(rounds, seed);
}