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
        unsigned long _byte = (regnum)*HLL_BITS / 8;                           \
        unsigned long _fb = (regnum)*HLL_BITS & 7;                             \
        unsigned long _fb8 = 8 - _fb;                                          \
        unsigned long _v = (val);                                              \
        _p[_byte] &= ~(HLL_REGISTER_MAX << _fb);                               \
        _p[_byte] |= _v << _fb;                                                \
        _p[_byte + 1] &= ~(HLL_REGISTER_MAX >> _fb8);                          \
        _p[_byte + 1] |= _v >> _fb8;                                           \
    } while (0)

#define TARGET_DEFAULT __attribute__((target("default")))
#define TARGET_AVX2 __attribute__((target("avx2")))
#define TARGET_AVX512 __attribute__((target("avx512f,avx512bw")))

void merge_base(uint8_t *reg_raw, const uint8_t *reg_dense) {
    uint8_t val;
    for (int i = 0; i < HLL_REGISTERS; i++) {
        HLL_DENSE_GET_REGISTER(val, reg_dense, i);
        if (val > reg_raw[i]) {
            reg_raw[i] = val;
        }
    }
}

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
void merge_avx2_1(uint8_t *reg_raw, const uint8_t *reg_dense) {
    const uint8_t *r = reg_dense - 4;
    const uint8_t *t = reg_raw;

    for (int i = 0; i < HLL_REGISTERS / 32; ++i) {
        __m256i x0, x;
        x0 = _mm256_loadu_si256((__m256i *)r);
        x = _mm256_shuffle_epi8(x0, avx2_shuffle);

        __m256i a1, a2, a3, a4;
        a1 = _mm256_and_si256(x, _mm256_set1_epi32(0x0000003f));
        a2 = _mm256_and_si256(x, _mm256_set1_epi32(0x00000fc0));
        a3 = _mm256_and_si256(x, _mm256_set1_epi32(0x0003f000));
        a4 = _mm256_and_si256(x, _mm256_set1_epi32(0x00fc0000));

        a2 = _mm256_slli_epi32(a2, 2);
        a3 = _mm256_slli_epi32(a3, 4);
        a4 = _mm256_slli_epi32(a4, 6);

        __m256i y1, y2, y;
        y1 = _mm256_or_si256(a1, a2);
        y2 = _mm256_or_si256(a3, a4);
        y = _mm256_or_si256(y1, y2);

        __m256i z = _mm256_loadu_si256((__m256i *)t);

        z = _mm256_max_epu8(z, y);

        _mm256_storeu_si256((__m256i *)t, z);

        r += 24;
        t += 32;
    }
}

void merge_avx2_2(uint8_t *reg_raw, const uint8_t *reg_dense) {
    uint8_t val;
    for (int i = 0; i < 32; i++) {
        HLL_DENSE_GET_REGISTER(val, reg_dense, i);
        if (val > reg_raw[i]) {
            reg_raw[i] = val;
        }
    }

    const uint8_t *r = reg_dense + 24 - 4;
    const uint8_t *t = reg_raw + 32;

    for (int i = 1; i < HLL_REGISTERS / 32; ++i) {
        __m256i x0, x;
        x0 = _mm256_loadu_si256((__m256i *)r);
        x = _mm256_shuffle_epi8(x0, avx2_shuffle);

        __m256i a1, a2, a3, a4;
        a1 = _mm256_and_si256(x, _mm256_set1_epi32(0x0000003f));
        a2 = _mm256_and_si256(x, _mm256_set1_epi32(0x00000fc0));
        a3 = _mm256_and_si256(x, _mm256_set1_epi32(0x0003f000));
        a4 = _mm256_and_si256(x, _mm256_set1_epi32(0x00fc0000));

        a2 = _mm256_slli_epi32(a2, 2);
        a3 = _mm256_slli_epi32(a3, 4);
        a4 = _mm256_slli_epi32(a4, 6);

        __m256i y1, y2, y;
        y1 = _mm256_or_si256(a1, a2);
        y2 = _mm256_or_si256(a3, a4);
        y = _mm256_or_si256(y1, y2);

        __m256i z = _mm256_loadu_si256((__m256i *)t);

        z = _mm256_max_epu8(z, y);

        _mm256_storeu_si256((__m256i *)t, z);

        r += 24;
        t += 32;
    }
}

void merge_avx2_3(uint8_t *reg_raw, const uint8_t *reg_dense) {
    uint8_t val;
    for (int i = 0; i < 8; i++) {
        HLL_DENSE_GET_REGISTER(val, reg_dense, i);
        if (val > reg_raw[i]) {
            reg_raw[i] = val;
        }
    }

    const uint8_t *r = reg_dense + 6 - 4;
    uint8_t *t = reg_raw + 8;

    for (int i = 0; i < HLL_REGISTERS / 32 - 1; ++i) {
        __m256i x0, x;
        x0 = _mm256_loadu_si256((__m256i *)r);
        x = _mm256_shuffle_epi8(x0, avx2_shuffle);

        __m256i a1, a2, a3, a4;
        a1 = _mm256_and_si256(x, _mm256_set1_epi32(0x0000003f));
        a2 = _mm256_and_si256(x, _mm256_set1_epi32(0x00000fc0));
        a3 = _mm256_and_si256(x, _mm256_set1_epi32(0x0003f000));
        a4 = _mm256_and_si256(x, _mm256_set1_epi32(0x00fc0000));

        a2 = _mm256_slli_epi32(a2, 2);
        a3 = _mm256_slli_epi32(a3, 4);
        a4 = _mm256_slli_epi32(a4, 6);

        __m256i y1, y2, y;
        y1 = _mm256_or_si256(a1, a2);
        y2 = _mm256_or_si256(a3, a4);
        y = _mm256_or_si256(y1, y2);

        __m256i z = _mm256_loadu_si256((__m256i *)t);

        z = _mm256_max_epu8(z, y);

        _mm256_storeu_si256((__m256i *)t, z);

        r += 24;
        t += 32;
    }

    for (int i = HLL_REGISTERS - 24; i < HLL_REGISTERS; i++) {
        HLL_DENSE_GET_REGISTER(val, reg_dense, i);
        if (val > reg_raw[i]) {
            reg_raw[i] = val;
        }
    }
}

#ifndef NO_AVX512

void merge_avx512_1(uint8_t *reg_raw, const uint8_t *reg_dense) {
    const __m512i shuffle = _mm512_set_epi8( //
        0x80, 11, 10, 9, 0x80, 8, 7, 6,      //
        0x80, 5, 4, 3, 0x80, 2, 1, 0,        //
        0x80, 15, 14, 13, 0x80, 12, 11, 10,  //
        0x80, 9, 8, 7, 0x80, 6, 5, 4,        //
        0x80, 11, 10, 9, 0x80, 8, 7, 6,      //
        0x80, 5, 4, 3, 0x80, 2, 1, 0,        //
        0x80, 15, 14, 13, 0x80, 12, 11, 10,  //
        0x80, 9, 8, 7, 0x80, 6, 5, 4         //
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

#endif

TARGET_DEFAULT
static void merge_dynamic_impl(uint8_t *reg_raw, const uint8_t *reg_dense) {
    merge_base(reg_raw, reg_dense);
}

TARGET_AVX2
static void merge_dynamic_impl(uint8_t *reg_raw, const uint8_t *reg_dense) {
    merge_avx2_3(reg_raw, reg_dense);
}

void merge_dynamic(uint8_t *reg_raw, const uint8_t *reg_dense) {
    merge_dynamic_impl(reg_raw, reg_dense);
}

void compress_base(uint8_t *reg_dense, const uint8_t *reg_raw) {
    for (int i = 0; i < HLL_REGISTERS; i++) {
        HLL_DENSE_SET_REGISTER(reg_dense, i, reg_raw[i]);
    }
}

void compress_avx2_1(uint8_t *reg_dense, const uint8_t *reg_raw) {
    const __m256i shuffle = _mm256_setr_epi8( //
        0, 1, 2,                              //
        4, 5, 6,                              //
        8, 9, 10,                             //
        12, 13, 14,                           //
        0x80, 0x80, 0x80, 0x80,               //
        0, 1, 2,                              //
        4, 5, 6,                              //
        8, 9, 10,                             //
        12, 13, 14,                           //
        0x80, 0x80, 0x80, 0x80                //
    );
    const uint8_t *r = reg_raw;
    uint8_t *t = reg_dense;

    for (int i = 0; i < HLL_REGISTERS / 32; ++i) {
        __m256i x = _mm256_loadu_si256((__m256i *)r);

        __m256i a1, a2, a3, a4;
        a1 = _mm256_and_si256(x, _mm256_set1_epi32(0x0000003f));
        a2 = _mm256_and_si256(x, _mm256_set1_epi32(0x00003f00));
        a3 = _mm256_and_si256(x, _mm256_set1_epi32(0x003f0000));
        a4 = _mm256_and_si256(x, _mm256_set1_epi32(0x3f000000));

        a2 = _mm256_srli_epi32(a2, 2);
        a3 = _mm256_srli_epi32(a3, 4);
        a4 = _mm256_srli_epi32(a4, 6);

        __m256i y1, y2, y;
        y1 = _mm256_or_si256(a1, a2);
        y2 = _mm256_or_si256(a3, a4);
        y = _mm256_or_si256(y1, y2);
        y = _mm256_shuffle_epi8(y, shuffle);

        __m128i lower, higher;
        lower = _mm256_castsi256_si128(y);
        higher = _mm256_extracti128_si256(y, 1);

        _mm_storeu_si128((__m128i *)t, lower);
        _mm_storeu_si128((__m128i *)(t + 12), higher);

        r += 32;
        t += 24;
    }
}

void compress_avx2_2(uint8_t *reg_dense, const uint8_t *reg_raw) {
    const __m256i shuffle = _mm256_setr_epi8( //
        0, 1, 2,                              //
        4, 5, 6,                              //
        8, 9, 10,                             //
        12, 13, 14,                           //
        0x80, 0x80, 0x80, 0x80,               //
        0, 1, 2,                              //
        4, 5, 6,                              //
        8, 9, 10,                             //
        12, 13, 14,                           //
        0x80, 0x80, 0x80, 0x80                //
    );
    const uint8_t *r = reg_raw;
    uint8_t *t = reg_dense;

    for (int i = 0; i < HLL_REGISTERS / 32 - 1; ++i) {
        __m256i x = _mm256_loadu_si256((__m256i *)r);

        __m256i a1, a2, a3, a4;
        a1 = _mm256_and_si256(x, _mm256_set1_epi32(0x0000003f));
        a2 = _mm256_and_si256(x, _mm256_set1_epi32(0x00003f00));
        a3 = _mm256_and_si256(x, _mm256_set1_epi32(0x003f0000));
        a4 = _mm256_and_si256(x, _mm256_set1_epi32(0x3f000000));

        a2 = _mm256_srli_epi32(a2, 2);
        a3 = _mm256_srli_epi32(a3, 4);
        a4 = _mm256_srli_epi32(a4, 6);

        __m256i y1, y2, y;
        y1 = _mm256_or_si256(a1, a2);
        y2 = _mm256_or_si256(a3, a4);
        y = _mm256_or_si256(y1, y2);
        y = _mm256_shuffle_epi8(y, shuffle);

        __m128i lower, higher;
        lower = _mm256_castsi256_si128(y);
        higher = _mm256_extracti128_si256(y, 1);

        _mm_storeu_si128((__m128i *)t, lower);
        _mm_storeu_si128((__m128i *)(t + 12), higher);

        r += 32;
        t += 24;
    }

    for (int i = HLL_REGISTERS - 32; i < HLL_REGISTERS; i++) {
        HLL_DENSE_SET_REGISTER(reg_dense, i, reg_raw[i]);
    }
}

#ifndef NO_AVX512
void compress_avx512_1(uint8_t *reg_dense, const uint8_t *reg_raw) {
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

void compress_avx512_2(uint8_t *reg_dense, const uint8_t *reg_raw) {
    const __m512i shuffle = _mm512_set_epi8( //
        0x80, 0x80, 0x80, 0x80,              //
        14, 13, 12,                          //
        10, 9, 8,                            //
        6, 5, 4,                             //
        2, 1, 0,                             //
        0x80, 0x80, 0x80, 0x80,              //
        14, 13, 12,                          //
        10, 9, 8,                            //
        6, 5, 4,                             //
        2, 1, 0,                             //
        0x80, 0x80, 0x80, 0x80,              //
        14, 13, 12,                          //
        10, 9, 8,                            //
        6, 5, 4,                             //
        2, 1, 0,                             //
        0x80, 0x80, 0x80, 0x80,              //
        14, 13, 12,                          //
        10, 9, 8,                            //
        6, 5, 4,                             //
        2, 1, 0                              //
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
        y = _mm512_shuffle_epi8(y, shuffle);

        __m128i p1, p2, p3, p4;
        p1 = _mm512_extracti64x2_epi64(y, 0);
        p2 = _mm512_extracti64x2_epi64(y, 1);
        p3 = _mm512_extracti64x2_epi64(y, 2);
        p4 = _mm512_extracti64x2_epi64(y, 3);

        _mm_storeu_si128((__m128i *)t, p1);
        _mm_storeu_si128((__m128i *)(t + 12), p2);
        _mm_storeu_si128((__m128i *)(t + 24), p3);
        _mm_storeu_si128((__m128i *)(t + 36), p4);

        r += 64;
        t += 48;
    }
}
#endif

TARGET_DEFAULT
static void compress_dynamic_impl(uint8_t *reg_dense, const uint8_t *reg_raw) {
    compress_base(reg_dense, reg_raw);
}

TARGET_AVX2
static void compress_dynamic_impl(uint8_t *reg_dense, const uint8_t *reg_raw) {
    compress_avx2_2(reg_dense, reg_raw);
}

void compress_dynamic(uint8_t *reg_dense, const uint8_t *reg_raw) {
    compress_dynamic_impl(reg_dense, reg_raw);
}

#define HLL_DENSE_REG_LEN (HLL_REGISTERS * HLL_BITS / 8)

void histogram_base_0(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense;
    for (int i = 0; i < HLL_REGISTERS; ++i) {
        uint8_t val;
        HLL_DENSE_GET_REGISTER(val, r, i);
        hist[val]++;
    }
}

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

/**
load
{????|AAAB|BBCC|CDDD|EEEF|FFGG|GHHH|????}

shuffle
{bbaaaaaa|ccccbbbb|ddddddcc|00000000} x8

and
{00aaaaaa|00000000|dddddd00|00000000} x8
mullo epi16 (<<0, <<6)
{00aaaaaa|00000000|00000000|00dddddd} x8

slli epi32 (<<2)
{aaaaaa00|ccbbbbbb|ddddcccc|000000dd} x8
slli epi32 (<<4)
{aaaa0000|bbbbbbaa|ddcccccc|0000dddd} x8
blend epi16
{aaaaaa00|00bbbbbb|ddcccccc|0000dddd} x8
and
{00000000|00bbbbbb|00cccccc|00000000} x8

or
{00aaaaaa|00bbbbbb|00cccccc|00dddddd} x8

 */
void histogram_avx2_2(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense - 4;

    alignas(32) uint8_t t[32];

    for (int j = 0; j < 512; ++j) {
        __m256i x0 = _mm256_loadu_si256((__m256i *)r);
        __m256i x1 = _mm256_shuffle_epi8(x0, avx2_shuffle);

#ifdef DEBUG
        alignas(32) uint8_t dbg[32];
        _mm256_store_si256((__m256i *)dbg, x1);
        int k = 4;
        for (int i = 0; i < 8; ++i) {
            // fprintf(stderr, "i = %d, k = %d\n", i, k);
            assert(dbg[i * 4 + 0] == r[k++]);
            assert(dbg[i * 4 + 1] == r[k++]);
            assert(dbg[i * 4 + 2] == r[k++]);
            assert(dbg[i * 4 + 3] == 0);
        }
#endif

        __m256i p1 = _mm256_and_si256(x1, _mm256_set1_epi32(0x00fc003f));
        __m256i a1 = _mm256_mullo_epi16(p1, _mm256_set1_epi32(0x00400001));

        __m256i p2 = _mm256_slli_epi32(x1, 2);
        __m256i p3 = _mm256_slli_epi32(x1, 4);
        __m256i b2 = _mm256_blend_epi16(p2, p3, 0b10101010);
        __m256i a2 = _mm256_and_si256(b2, _mm256_set1_epi32(0x003f3f00));

        __m256i y = _mm256_or_si256(a1, a2);

        _mm256_store_si256((__m256i *)t, y);

#ifdef DEBUG
        _mm256_store_si256((__m256i *)dbg, a1);
        for (int i = 0; i < 32; i += 4) {
            assert(dbg[i + 1] == 0);
            assert(dbg[i + 2] == 0);
        }
        _mm256_store_si256((__m256i *)dbg, a2);
        for (int i = 0; i < 32; i += 4) {
            assert(dbg[i + 0] == 0);
            assert(dbg[i + 3] == 0);
        }

        for (int i = 0; i < 32; ++i) {
            assert(t[i] < 64);
            uint8_t val;
            HLL_DENSE_GET_REGISTER(val, reg_dense, (j * 32 + i));
            // fprintf(stderr, "%d, %d, %d\n", i, t[i], val);
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

void histogram_avx2_3(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense - 4;

    alignas(32) uint8_t vh[16][64];
    alignas(32) uint8_t t[32];
    alignas(32) int h[64];

    memset(vh, 0, sizeof(vh));

    for (int j = 0; j < 512; ++j) {
        __m256i x0 = _mm256_loadu_si256((__m256i *)r);
        __m256i x1 = _mm256_shuffle_epi8(x0, avx2_shuffle);

#ifdef DEBUG
        alignas(32) uint8_t dbg[32];
        _mm256_store_si256((__m256i *)dbg, x1);
        int k = 4;
        for (int i = 0; i < 8; ++i) {
            // fprintf(stderr, "i = %d, k = %d\n", i, k);
            assert(dbg[i * 4 + 0] == r[k++]);
            assert(dbg[i * 4 + 1] == r[k++]);
            assert(dbg[i * 4 + 2] == r[k++]);
            assert(dbg[i * 4 + 3] == 0);
        }
#endif

        __m256i p1 = _mm256_and_si256(x1, _mm256_set1_epi32(0x00fc003f));
        __m256i a1 = _mm256_mullo_epi16(p1, _mm256_set1_epi32(0x00400001));

        __m256i p2 = _mm256_slli_epi32(x1, 2);
        __m256i p3 = _mm256_slli_epi32(x1, 4);
        __m256i b2 = _mm256_blend_epi16(p2, p3, 0b10101010);
        __m256i a2 = _mm256_and_si256(b2, _mm256_set1_epi32(0x003f3f00));

        __m256i y = _mm256_or_si256(a1, a2);

        _mm256_store_si256((__m256i *)t, y);

#ifdef DEBUG
        _mm256_store_si256((__m256i *)dbg, a1);
        for (int i = 0; i < 32; i += 4) {
            assert(dbg[i + 1] == 0);
            assert(dbg[i + 2] == 0);
        }
        _mm256_store_si256((__m256i *)dbg, a2);
        for (int i = 0; i < 32; i += 4) {
            assert(dbg[i + 0] == 0);
            assert(dbg[i + 3] == 0);
        }

        for (int i = 0; i < 32; ++i) {
            assert(t[i] < 64);
            uint8_t val;
            HLL_DENSE_GET_REGISTER(val, reg_dense, (j * 32 + i));
            // fprintf(stderr, "%d, %d, %d\n", i, t[i], val);
            assert(t[i] == val);
        }
#endif
        vh[0][t[0]]++, vh[1][t[1]]++, vh[2][t[2]]++, vh[3][t[3]]++;
        vh[4][t[4]]++, vh[5][t[5]]++, vh[6][t[6]]++, vh[7][t[7]]++;
        vh[8][t[8]]++, vh[9][t[9]]++, vh[10][t[10]]++, vh[11][t[11]]++;
        vh[12][t[12]]++, vh[13][t[13]]++, vh[14][t[14]]++, vh[15][t[15]]++;

        vh[0][t[16]]++, vh[1][t[17]]++, vh[2][t[18]]++, vh[3][t[19]]++;
        vh[4][t[20]]++, vh[5][t[21]]++, vh[6][t[22]]++, vh[7][t[23]]++;
        vh[8][t[24]]++, vh[9][t[25]]++, vh[10][t[26]]++, vh[11][t[27]]++;
        vh[12][t[28]]++, vh[13][t[29]]++, vh[14][t[30]]++, vh[15][t[31]]++;

        r += 24;
    }
    memcpy(h, hist, sizeof(h));
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 64; j++) {
            h[j] += vh[i][j];
        }
    }
    memcpy(hist, h, sizeof(h));
}

const __m256i avx2_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

#ifndef NO_AVX512

void histogram_avx512_1(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense - 4;

    alignas(64) int vbins[64 * 8];
    memset(vbins, 0, sizeof(vbins));

    for (int j = 0; j < 512; ++j) {
        __m256i x0 = _mm256_loadu_si256((__m256i *)r);
        __m256i x1 = _mm256_shuffle_epi8(x0, avx2_shuffle);

        __m256i a1, a2, a3, a4;
        a1 = _mm256_and_si256(x1, _mm256_set1_epi32(0x0000003f));
        a2 = _mm256_and_si256(x1, _mm256_set1_epi32(0x00000fc0));
        a3 = _mm256_and_si256(x1, _mm256_set1_epi32(0x0003f000));
        a4 = _mm256_and_si256(x1, _mm256_set1_epi32(0x00fc0000));

        a1 = _mm256_slli_epi32(a1, 3);
        a2 = _mm256_srli_epi32(a2, 6 - 3);
        a3 = _mm256_srli_epi32(a3, 12 - 3);
        a4 = _mm256_srli_epi32(a4, 18 - 3);

        a1 = _mm256_add_epi32(a1, avx2_indices);
        a2 = _mm256_add_epi32(a2, avx2_indices);
        a3 = _mm256_add_epi32(a3, avx2_indices);
        a4 = _mm256_add_epi32(a4, avx2_indices);

#ifdef DEBUG
        {
            int dbg[4][8];
            _mm256_storeu_si256((__m256i *)(&dbg[0][0]), a1);
            _mm256_storeu_si256((__m256i *)(&dbg[1][0]), a2);
            _mm256_storeu_si256((__m256i *)(&dbg[2][0]), a3);
            _mm256_storeu_si256((__m256i *)(&dbg[3][0]), a4);
            for (int i = 0; i < 4; ++i) {
                for (int k = 0; k < 8; ++k) {
                    uint8_t val;
                    HLL_DENSE_GET_REGISTER(val, reg_dense,
                                           (j * 32 + k * 4 + i));
                    int idx = int(val) * 8 + k;
                    if (dbg[i][k] != idx) {
                        fprintf(stderr, "j=%d k=%d i=%d val=%d idx=%d dbg=%d\n",
                                j, k, i, val, idx, dbg[i][k]);
                    }
                    assert(dbg[i][k] == idx);

                    // vbins[idx] += 1;
                }
            }
        }
#endif

        __m256i h1, h2, h3, h4;
        h1 = _mm256_i32gather_epi32(vbins, a1, 4);
        h1 = _mm256_add_epi32(h1, _mm256_set1_epi32(1));
        _mm256_i32scatter_epi32(vbins, a1, h1, 4);

        h2 = _mm256_i32gather_epi32(vbins, a2, 4);
        h2 = _mm256_add_epi32(h2, _mm256_set1_epi32(1));
        _mm256_i32scatter_epi32(vbins, a2, h2, 4);

        h3 = _mm256_i32gather_epi32(vbins, a3, 4);
        h3 = _mm256_add_epi32(h3, _mm256_set1_epi32(1));
        _mm256_i32scatter_epi32(vbins, a3, h3, 4);

        h4 = _mm256_i32gather_epi32(vbins, a4, 4);
        h4 = _mm256_add_epi32(h4, _mm256_set1_epi32(1));
        _mm256_i32scatter_epi32(vbins, a4, h4, 4);

        r += 24;
    }
    for (int i = 0; i < 64; ++i) {
        int sum = 0;
        for (int j = 0; j < 8; ++j) {
            sum += vbins[i * 8 + j];
        }
        hist[i] += sum;
    }
}

void histogram_avx512_2(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense - 4;

    alignas(64) int vbins[64 * 16];
    memset(vbins, 0, sizeof(vbins));

    for (int j = 0; j < 512; ++j) {
        __m256i x0 = _mm256_loadu_si256((__m256i *)r);
        __m256i x1 = _mm256_shuffle_epi8(x0, avx2_shuffle);

        __m256i p1 = _mm256_and_si256(x1, _mm256_set1_epi32(0x00fc003f));
        __m256i a1 = _mm256_mullo_epi16(p1, _mm256_set1_epi32(0x00400001));

        __m256i p2 = _mm256_slli_epi32(x1, 2);
        __m256i p3 = _mm256_slli_epi32(x1, 4);
        __m256i b2 = _mm256_blend_epi16(p2, p3, 0b10101010);
        __m256i a2 = _mm256_and_si256(b2, _mm256_set1_epi32(0x003f3f00));

        __m256i y = _mm256_or_si256(a1, a2);

        const __m512i indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                  10, 11, 12, 13, 14, 15);

        __m512i i0, i1, y0, y1;
        i0 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(y, 0));
        i0 = _mm512_add_epi32(_mm512_slli_epi32(i0, 4), indices);
        y0 = _mm512_i32gather_epi32(i0, vbins, 4);
        y0 = _mm512_add_epi32(y0, _mm512_set1_epi32(1));
        _mm512_i32scatter_epi32(vbins, i0, y0, 4);

        i1 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(y, 1));
        i1 = _mm512_add_epi32(_mm512_slli_epi32(i1, 4), indices);
        y1 = _mm512_i32gather_epi32(i1, vbins, 4);
        y1 = _mm512_add_epi32(y1, _mm512_set1_epi32(1));
        _mm512_i32scatter_epi32(vbins, i1, y1, 4);

        r += 24;
    }

    for (int i = 0; i < 64; ++i) {
        hist[i] += _mm512_reduce_add_epi32(_mm512_load_si512(vbins + i * 16));
    }
}

void histogram_avx512_3(const uint8_t *reg_dense, int *hist) {
    const uint8_t *r = reg_dense - 4;

    alignas(64) int vbins[64 * 16];
    memset(vbins, 0, sizeof(vbins));

    for (int j = 0; j < 256; ++j) {
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

        __m256i x0 = _mm256_loadu_si256((__m256i *)r);
        __m256i x1 = _mm256_loadu_si256((__m256i *)(r + 24));
        __m512i x = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);
        x = _mm512_shuffle_epi8(x, shuffle);

#ifdef DEBUG
        {
            uint8_t dbg[64];
            _mm512_storeu_si512((__m512i *)dbg, x);
            int k = 4;
            for (int i = 0; i < 16; ++i) {
                // fprintf(stderr, "i = %d, k = %d\n", i, k);
                assert(dbg[i * 4 + 0] == r[k++]);
                assert(dbg[i * 4 + 1] == r[k++]);
                assert(dbg[i * 4 + 2] == r[k++]);
                assert(dbg[i * 4 + 3] == 0);
            }
        }
#endif

        __m512i a1, a2, a3, a4;
        a1 = _mm512_and_si512(x, _mm512_set1_epi32(0x0000003f));
        a2 = _mm512_and_si512(x, _mm512_set1_epi32(0x00000fc0));
        a3 = _mm512_and_si512(x, _mm512_set1_epi32(0x0003f000));
        a4 = _mm512_and_si512(x, _mm512_set1_epi32(0x00fc0000));

#ifdef DEBUG
        {
            int dbg[4][16];
            _mm512_storeu_si512((__m512i *)(&dbg[0][0]), a1);
            _mm512_storeu_si512((__m512i *)(&dbg[1][0]), a2);
            _mm512_storeu_si512((__m512i *)(&dbg[2][0]), a3);
            _mm512_storeu_si512((__m512i *)(&dbg[3][0]), a4);
            for (int i = 0; i < 4; ++i) {
                for (int k = 0; k < 16; ++k) {
                    uint8_t val;
                    HLL_DENSE_GET_REGISTER(val, reg_dense,
                                           (j * 64 + k * 4 + i));
                    int idx = int(val) << (6 * i);
                    if (dbg[i][k] != idx) {
                        fprintf(stderr, "j=%d k=%d i=%d val=%d idx=%d dbg=%d\n",
                                j, k, i, val, idx, dbg[i][k]);
                    }
                    assert(dbg[i][k] == idx);
                }
            }
        }
#endif

        a1 = _mm512_slli_epi32(a1, 4);
        a2 = _mm512_srli_epi32(a2, 6 - 4);
        a3 = _mm512_srli_epi32(a3, 12 - 4);
        a4 = _mm512_srli_epi32(a4, 18 - 4);

#ifdef DEBUG
        {
            int dbg[4][16];
            _mm512_storeu_si512((__m512i *)(&dbg[0][0]), a1);
            _mm512_storeu_si512((__m512i *)(&dbg[1][0]), a2);
            _mm512_storeu_si512((__m512i *)(&dbg[2][0]), a3);
            _mm512_storeu_si512((__m512i *)(&dbg[3][0]), a4);
            for (int i = 0; i < 4; ++i) {
                for (int k = 0; k < 16; ++k) {
                    uint8_t val;
                    HLL_DENSE_GET_REGISTER(val, reg_dense,
                                           (j * 64 + k * 4 + i));
                    int idx = int(val) * 16;
                    if (dbg[i][k] != idx) {
                        fprintf(stderr, "j=%d k=%d i=%d val=%d idx=%d dbg=%d\n",
                                j, k, i, val, idx, dbg[i][k]);
                    }
                    assert(dbg[i][k] == idx);
                }
            }
        }
#endif

        const __m512i indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                  10, 11, 12, 13, 14, 15);

        a1 = _mm512_add_epi32(a1, indices);
        a2 = _mm512_add_epi32(a2, indices);
        a3 = _mm512_add_epi32(a3, indices);
        a4 = _mm512_add_epi32(a4, indices);

        __m512i h1, h2, h3, h4;
        h1 = _mm512_i32gather_epi32(a1, vbins, 4);
        h1 = _mm512_add_epi32(h1, _mm512_set1_epi32(1));
        _mm512_i32scatter_epi32(vbins, a1, h1, 4);

        h2 = _mm512_i32gather_epi32(a2, vbins, 4);
        h2 = _mm512_add_epi32(h2, _mm512_set1_epi32(1));
        _mm512_i32scatter_epi32(vbins, a2, h2, 4);

        h3 = _mm512_i32gather_epi32(a3, vbins, 4);
        h3 = _mm512_add_epi32(h3, _mm512_set1_epi32(1));
        _mm512_i32scatter_epi32(vbins, a3, h3, 4);

        h4 = _mm512_i32gather_epi32(a4, vbins, 4);
        h4 = _mm512_add_epi32(h4, _mm512_set1_epi32(1));
        _mm512_i32scatter_epi32(vbins, a4, h4, 4);

#ifdef DEBUG
        {
            int dbg[4][16];
            _mm512_storeu_si512((__m512i *)(&dbg[0][0]), a1);
            _mm512_storeu_si512((__m512i *)(&dbg[1][0]), a2);
            _mm512_storeu_si512((__m512i *)(&dbg[2][0]), a3);
            _mm512_storeu_si512((__m512i *)(&dbg[3][0]), a4);
            for (int i = 0; i < 4; ++i) {
                for (int k = 0; k < 16; ++k) {
                    uint8_t val;
                    HLL_DENSE_GET_REGISTER(val, reg_dense,
                                           (j * 64 + k * 4 + i));
                    int idx = int(val) * 16 + k;
                    if (dbg[i][k] != idx) {
                        fprintf(stderr, "j=%d k=%d i=%d val=%d idx=%d dbg=%d\n",
                                j, k, i, val, idx, dbg[i][k]);
                    }
                    assert(dbg[i][k] == val * 16 + k);

                    // vbins[idx] += 1;
                }
            }
        }
#endif

        r += 48;
    }

    for (int i = 0; i < 64; ++i) {
        hist[i] += _mm512_reduce_add_epi32(_mm512_load_si512(vbins + i * 16));
    }
}

#endif

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
            printf("[%-18s]: %.6fs\n", names[i].c_str(), runtime[i]);
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

        std::vector<void (*)(uint8_t *, const uint8_t *)> funcs{
            merge_avx2_1, //
            merge_avx2_2, //
            merge_avx2_3, //
#ifndef NO_AVX512
            merge_avx512_1, //
            merge_avx512_2, //
#endif
            merge_dynamic,
        };

        int num = funcs.size();
        for (int j = 0; j < num; ++j) {
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
    group.add("merge_avx2_1", [=]() {
        memset(reg_raw, 0, HLL_REGISTERS);
        merge_avx2_1(reg_raw, reg_dense);
    });
    group.add("merge_avx2_2", [=]() {
        memset(reg_raw, 0, HLL_REGISTERS);
        merge_avx2_2(reg_raw, reg_dense);
    });
    group.add("merge_avx2_3", [=]() {
        memset(reg_raw, 0, HLL_REGISTERS);
        merge_avx2_3(reg_raw, reg_dense);
    });
#ifndef NO_AVX512
    group.add("merge_avx512_1", [=]() {
        memset(reg_raw, 0, HLL_REGISTERS);
        merge_avx512_1(reg_raw, reg_dense);
    });
    group.add("merge_avx512_2", [=]() {
        memset(reg_raw, 0, HLL_REGISTERS);
        merge_avx512_2(reg_raw, reg_dense);
    });
#endif
    group.add("merge_dynamic", [=]() {
        memset(reg_raw, 0, HLL_REGISTERS);
        merge_dynamic(reg_raw, reg_dense);
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

        std::vector<void (*)(uint8_t *, const uint8_t *)> funcs{
            compress_avx2_1, //
            compress_avx2_2, //
#ifndef NO_AVX512
            compress_avx512_1, //
            compress_avx512_2, //
#endif
            compress_dynamic,
        };

        int num = funcs.size();
        for (int j = 0; j < num; ++j) {
            memset(buf4, 0, sizeof(buf4));
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
    group.add("compress_avx2_1", [=]() {
        compress_avx2_1(reg_dense, reg_raw); //
    });
    group.add("compress_avx2_2", [=]() {
        compress_avx2_2(reg_dense, reg_raw); //
    });
#ifndef NO_AVX512
    group.add("compress_avx512_1", [=]() {
        compress_avx512_1(reg_dense, reg_raw); //
    });
    group.add("compress_avx512_2", [=]() {
        compress_avx512_1(reg_dense, reg_raw); //
    });
#endif
    group.add("compress_dynamic", [=]() {
        compress_dynamic(reg_dense, reg_raw); //
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
        histogram_base_0(reg_dense, hist1);

        std::vector<void (*)(const uint8_t *, int *)> funcs{
            histogram_base_1, //
            histogram_base_2, //
            histogram_unroll, //
            histogram_avx2_1, //
            histogram_avx2_2, //
            histogram_avx2_3, //
#ifndef NO_AVX512
            histogram_avx512_1, //
            histogram_avx512_2, //
            histogram_avx512_3, //
#endif
        };

        int num = funcs.size();
        for (int j = 0; j < num; ++j) {
            memset(hist2, 0, sizeof(hist2));
            funcs[j](reg_dense, hist2);

            int idx = check_histogram(hist1, hist2);
            // #ifdef DEBUG
            //             for (int i = 0; i < 64; i++) {
            //                 printf("%02d %3d %3d\n", i, hist1[i], hist2[i]);
            //             }
            //             printf("idx=%d\n", idx);
            // #endif
            if (idx >= 0) {
                fprintf(stderr, "error: %d, %d, %d, %d\n", j, idx, hist1[idx],
                        hist2[idx]);
                exit(1);
            }
        }
    }

    BenchmarkGroup group;
    group.add("histogram_base_0", [=]() {
        histogram_base_0(reg_dense, hist1); //
    });
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
    group.add("histogram_avx2_2", [=]() {
        histogram_avx2_2(reg_dense, hist1); //
    });
    group.add("histogram_avx2_3", [=]() {
        histogram_avx2_3(reg_dense, hist1); //
    });
#ifndef NO_AVX512
    group.add("histogram_avx512_1", [=]() {
        histogram_avx512_1(reg_dense, hist1); //
    });
    group.add("histogram_avx512_2", [=]() {
        histogram_avx512_2(reg_dense, hist1); //
    });
    group.add("histogram_avx512_3", [=]() {
        histogram_avx512_3(reg_dense, hist1); //
    });
#endif

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

// AVX512:
// g++ bench.cpp -O3 -march=native -Wall -Wextra -std=c++20 -o a.out && ./a.out
// | tee cpp_bench.log

// NO_AVX512:
// g++ bench.cpp -O3 -march=native -Wall -Wextra -std=c++20 -o a.out -DNO_AVX512
// && ./a.out | tee cpp_bench.log
