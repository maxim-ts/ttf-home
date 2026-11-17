// W8A8 CPU matmul for small batches of rows (ttf-home, 17 Nov 2026): a Q8_0 weight (blocks of 32, one fp16 scale
// each) is requantized once to int8 with one scale per output row, the activations to int8 with one scale per row
// (token), and every dot product runs over the whole K in integers (AVX-VNNI vpdpbusd, or AVX2 maddubs). ggml's Q8_0
// path converts to float after every 32-block, which bounds it at ~5 vector ops per 32 multiply-adds; this runs at ~1.
// VoxCPM2's DiT on a laptop i9-14900HX (M = 22 rows, AVX-VNNI): 3.5x ggml's Q8_0 per core (tile 4 x 3, 12 accumulators
// in named registers). The numerics differ from Q8_0 (row instead of block scales): on by TTF_W8A8=1.
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
#    define TTF_W8_VNNI 1
#    if defined(__AVXVNNI__)
#        define TTF_DPBUSD(acc, a, b) _mm256_dpbusd_avx_epi32(acc, a, b)
#    else
#        define TTF_DPBUSD(acc, a, b) _mm256_dpbusd_epi32(acc, a, b)
#    endif
#elif defined(__AVX2__)
#    define TTF_W8_VNNI 0
#endif

#if defined(TTF_W8_VNNI)

#ifndef TTF_W8_PD
#    define TTF_W8_PD 1
#endif

namespace {

struct W8 {
    int64_t              K = 0, N = 0;
    std::vector<int8_t>  q;   // [N, K]
    std::vector<float>   s;   // [N]
    std::vector<int32_t> rs;  // [N] row sums (VNNI: the u8 offset's correction)
};

std::mutex                                   g_mu;
std::unordered_map<const void *, W8 *>       g_cache;  // by the weight's data pointer (weights do not move)

W8 * convert(const ggml_tensor * w) {
    const int64_t K = w->ne[0], N = w->ne[1], KB = K / 32;
    W8 *          c = new W8;
    c->K = K;
    c->N = N;
    c->q.resize((size_t) N * K);
    c->s.resize(N);
    c->rs.resize(N);
    std::vector<float> f(K);
    for (int64_t n = 0; n < N; ++n) {
        const block_q8_0 * b = (const block_q8_0 *) ((const char *) w->data + n * w->nb[1]);
        float              amax = 0;
        for (int64_t j = 0; j < KB; ++j) {
            const float d = GGML_FP16_TO_FP32(b[j].d);
            for (int i = 0; i < 32; ++i) {
                f[j * 32 + i] = d * b[j].qs[i];
                amax          = std::fmax(amax, std::fabs(f[j * 32 + i]));
            }
        }
        const float s = amax / 127.f, is = s > 0 ? 1.f / s : 0.f;
        int32_t     sum = 0;
        for (int64_t k = 0; k < K; ++k) {
            const int q = (int) std::lrintf(f[k] * is);
            c->q[(size_t) n * K + k] = (int8_t) q;
            sum += q;
        }
        c->s[n]  = s;
        c->rs[n] = sum;
    }
    return c;
}

inline int hsum8i(__m256i v) {
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    a         = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4e));
    a         = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xb1));
    return _mm_cvtsi128_si32(a);
}

// one activation row to int8 codes with one scale (round to nearest even, as lrintf); VNNI: stored + 128 (u8)
void quant_row(const float * p, int64_t K, uint8_t * q, float * sd) {
    const __m256 sgn = _mm256_set1_ps(-0.0f);
    __m256       mx  = _mm256_setzero_ps();
    for (int64_t k = 0; k < K; k += 8) {
        mx = _mm256_max_ps(mx, _mm256_andnot_ps(sgn, _mm256_loadu_ps(p + k)));
    }
    __m128 h = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
    h        = _mm_max_ps(h, _mm_movehl_ps(h, h));
    h        = _mm_max_ss(h, _mm_movehdup_ps(h));
    const float d = _mm_cvtss_f32(h) / 127.f, id = d > 0 ? 1.f / d : 0.f;
    *sd           = d;
    const __m256  vi  = _mm256_set1_ps(id);
    const __m256i prm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t k = 0; k < K; k += 32) {
        const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + k), vi));
        const __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + k + 8), vi));
        const __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + k + 16), vi));
        const __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + k + 24), vi));
        __m256i       b  = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        b                = _mm256_permutevar8x32_epi32(b, prm);  // the packs interleave the 128-bit lanes
#if TTF_W8_VNNI
        b = _mm256_xor_si256(b, _mm256_set1_epi8((char) 0x80));  // + 128
#endif
        _mm256_storeu_si256((__m256i *) (q + k), b);
    }
}

// rows [n0, n1) of W x all M activation rows; tile 4 x 3 (12 accumulators, 3 weight + 1 activation vector: 16 ymm)
void gemm(const W8 * w, int64_t K, int64_t n0, int64_t n1, const uint8_t * aq, const float * sa, int64_t M, float * y,
          size_t ldy) {
    constexpr int MT = 4, NT = 3;
#if TTF_W8_VNNI
#    define TTF_W8_DOT(acc, a, w) TTF_DPBUSD(acc, a, w)
#else  // a holds signed codes: |w| x sign(a, w) stays within int16 pairs (|w|, |a| <= 127)
    const __m256i ones = _mm256_set1_epi16(1);
#    define TTF_W8_DOT(acc, a, w) \
        _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(a, w)), ones))
#endif
    for (int64_t n = n0; n < n1; n += NT) {
        const int      nn = (int) std::min<int64_t>(NT, n1 - n);
        const int8_t * wr[NT];
        for (int j = 0; j < NT; ++j) {
            wr[j] = w->q.data() + (size_t) (n + (j < nn ? j : 0)) * K;
        }
        // the next tile's weight rows on their way while this one computes (cold weights: DRAM-bound otherwise)
        constexpr int PD = TTF_W8_PD;  // tiles ahead
        if (n + PD * NT < n1) {
            const int8_t * nx = w->q.data() + (size_t) (n + PD * NT) * K;
            const int64_t  nb = std::min<int64_t>(NT, n1 - n - PD * NT) * K;
            for (int64_t o = 0; o < nb; o += 64) {
                _mm_prefetch((const char *) (nx + o), _MM_HINT_T0);
            }
        }
        for (int64_t m = 0; m < M; m += MT) {
            const int       mm = (int) std::min<int64_t>(MT, M - m);
            const uint8_t * ar[MT];
            for (int i = 0; i < MT; ++i) {
                ar[i] = aq + (size_t) (m + (i < mm ? i : 0)) * K;
            }
            // named accumulators: as an array GCC keeps them on the stack across the loop
            __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c10 = c00, c11 = c00, c12 = c00, c20 = c00, c21 = c00,
                    c22 = c00, c30 = c00, c31 = c00, c32 = c00;
            const int8_t * w0 = wr[0], * w1 = wr[1], * w2 = wr[2];
            const uint8_t * a0 = ar[0], * a1 = ar[1], * a2 = ar[2], * a3 = ar[3];
            for (int64_t k = 0; k < K; k += 32) {
                const __m256i v0 = _mm256_loadu_si256((const __m256i *) (w0 + k));
                const __m256i v1 = _mm256_loadu_si256((const __m256i *) (w1 + k));
                const __m256i v2 = _mm256_loadu_si256((const __m256i *) (w2 + k));
                __m256i       a  = _mm256_loadu_si256((const __m256i *) (a0 + k));
                c00 = TTF_W8_DOT(c00, a, v0);
                c01 = TTF_W8_DOT(c01, a, v1);
                c02 = TTF_W8_DOT(c02, a, v2);
                a   = _mm256_loadu_si256((const __m256i *) (a1 + k));
                c10 = TTF_W8_DOT(c10, a, v0);
                c11 = TTF_W8_DOT(c11, a, v1);
                c12 = TTF_W8_DOT(c12, a, v2);
                a   = _mm256_loadu_si256((const __m256i *) (a2 + k));
                c20 = TTF_W8_DOT(c20, a, v0);
                c21 = TTF_W8_DOT(c21, a, v1);
                c22 = TTF_W8_DOT(c22, a, v2);
                a   = _mm256_loadu_si256((const __m256i *) (a3 + k));
                c30 = TTF_W8_DOT(c30, a, v0);
                c31 = TTF_W8_DOT(c31, a, v1);
                c32 = TTF_W8_DOT(c32, a, v2);
            }
            const __m256i acc[MT][NT] = { { c00, c01, c02 }, { c10, c11, c12 }, { c20, c21, c22 }, { c30, c31, c32 } };
            for (int i = 0; i < mm; ++i) {
                for (int j = 0; j < nn; ++j) {
#if TTF_W8_VNNI
                    const int32_t d = hsum8i(acc[i][j]) - 128 * w->rs[n + j];
#else
                    const int32_t d = hsum8i(acc[i][j]);
#endif
                    y[(size_t) (m + i) * ldy + n + j] = w->s[n + j] * sa[m + i] * (float) d;
                }
            }
        }
    }
}

}  // namespace

extern "C" bool ggml_ttf_w8a8_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
extern "C" bool ggml_ttf_w8a8_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    static const int on = [] {
        const char * e = std::getenv("TTF_W8A8");
        return e ? std::atoi(e) : 0;
    }();
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];
    if (!on || params->use_ref || w->type != GGML_TYPE_Q8_0 || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t K = w->ne[0], N = w->ne[1], M = x->ne[1] * x->ne[2] * x->ne[3];  // x [K, T, B]: B T rows, one weight
    if (K % 32 || M > 64 || w->ne[2] != 1 || w->ne[3] != 1 || !ggml_is_contiguous(x) || !ggml_is_contiguous(w) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }
    // work buffer: codes [M, K] then scales [M]; the plan reserves the Q8_0 copy of x (M K 34/32 bytes), enough for K >= 64
    if (params->wsize < (size_t) (M * K + 4 * M + 64)) {
        return false;
    }
    const int ith = params->ith, nth = params->nth;
    // the weight in W8, converted once (thread 0; the others wait at the barrier)
    W8 * c = nullptr;
    if (ith == 0) {
        std::lock_guard<std::mutex> lk(g_mu);
        auto                        it = g_cache.find(w->data);
        if (it == g_cache.end() || it->second->K != K || it->second->N != N) {  // a freed weight's address reused
            if (it != g_cache.end()) {
                delete it->second;
                g_cache.erase(it);
            }
            g_cache.emplace(w->data, convert(w));
        }
    }
    uint8_t * aq = (uint8_t *) params->wdata;
    float *   sa = (float *) (aq + GGML_PAD(M * K, 32));
    for (int64_t m = ith; m < M; m += nth) {  // activations per row
        quant_row((const float *) x->data + m * K, K, aq + (size_t) m * K, sa + m);
    }
    ggml_barrier(params->threadpool);
    {
        std::lock_guard<std::mutex> lk(g_mu);
        c = g_cache.at(w->data);
    }
    const int64_t per = ((N + nth - 1) / nth + 2) / 3 * 3;
    const int64_t n0 = std::min<int64_t>(N, ith * per), n1 = std::min<int64_t>(N, n0 + per);
    if (n0 < n1) {
        gemm(c, K, n0, n1, aq, sa, M, (float *) dst->data, (size_t) N);
    }
    return true;
}

#else

extern "C" bool ggml_ttf_w8a8_mul_mat(const struct ggml_compute_params *, struct ggml_tensor *) {
    return false;
}

#endif
