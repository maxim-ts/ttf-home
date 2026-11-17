// CPU kernels for audio VAEs (ttf-home, 17 Nov 2026), used as ggml custom ops by the engine on CPU sessions:
//   ttf_cpu_snake   y = x + inv sin^2(alpha x), one pass (the graph form is 5 passes and a scalar sinf per element)
//   ttf_cpu_dwconv  depthwise dilated conv, stride 1, direct (the graph form is im2col, k x the input, plus a matmul)
// Layout ggml [T, C] (time fastest). Built with the CPU backend's flags, so the loops vectorize for the machine.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// sin^2(x) for |x| < ~1e5: Cody-Waite reduction by pi/2, then sin^2 or cos^2 of the remainder (the sign is not needed);
// branch-free so it vectorizes; |error| ~1e-7 (cephes sinf / cosf polynomials)
inline float sin2(float x) {
    const float j  = std::nearbyint(x * 0.63661977236758134f);
    float       r  = x - j * 1.5703125f;
    r              = r - j * 4.837512969970703125e-4f;
    r              = r - j * 7.54978995489188216e-8f;
    const float z  = r * r;
    const float s  = r + r * z * (-1.6666654611e-1f + z * (8.3321608736e-3f + z * -1.9515295891e-4f));
    const float c  = 1.0f - 0.5f * z + z * z * (4.166664568298827e-2f + z * (-1.388731625493765e-3f + z * 2.443315711809948e-5f));
    const int   q  = (int) j;
    const float v  = (q & 1) ? c : s;
    return v * v;
}

inline void row_f32(const ggml_tensor * t, int64_t i, float * out, int64_t n) {
    const char * p = (const char *) t->data + i * t->nb[1];
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out, p, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) p, out, n);
    } else {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) p, out, n);
    }
}

}  // namespace

// dst [T, C]; src0 x [T, C] f32 contiguous, src1 alpha [C], src2 inv [C] (f32 / f16 / bf16, any shape with C elements)
extern "C" void ttf_cpu_snake(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * x = dst->src[0];
    const int64_t       T = x->ne[0], C = ggml_nrows(x);
    std::vector<float>  al(C), iv(C);
    // the per-channel parameters, flattened (both tensors hold C elements in one row or a column)
    {
        std::vector<float> a(ggml_nelements(dst->src[1])), b(ggml_nelements(dst->src[2]));
        for (int64_t r = 0, o = 0; r < ggml_nrows(dst->src[1]); ++r, o += dst->src[1]->ne[0]) {
            row_f32(dst->src[1], r, a.data() + o, dst->src[1]->ne[0]);
        }
        for (int64_t r = 0, o = 0; r < ggml_nrows(dst->src[2]); ++r, o += dst->src[2]->ne[0]) {
            row_f32(dst->src[2], r, b.data() + o, dst->src[2]->ne[0]);
        }
        al.swap(a);
        iv.swap(b);
    }
    const int64_t per = (C + nth - 1) / nth, c0 = ith * per, c1 = std::min(C, c0 + per);
    for (int64_t c = c0; c < c1; ++c) {
        const float * xr = (const float *) ((const char *) x->data + c * x->nb[1]);
        float *       yr = (float *) ((char *) dst->data + c * dst->nb[1]);
        const float   a = al[c], v = iv[c];
        for (int64_t t = 0; t < T; ++t) {
            yr[t] = xr[t] + v * sin2(a * xr[t]);
        }
    }
}

// dst [Tout, C] = sum_j w[j, c] x[t + j d, c]; src0 x [Tin, C] f32 (rows contiguous), src1 w [k, 1, C]; userdata = d
extern "C" void ttf_cpu_dwconv(ggml_tensor * dst, int ith, int nth, void * ud) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const int64_t       d = (int64_t) (intptr_t) ud, k = w->ne[0], C = dst->ne[1], To = dst->ne[0];
    const int64_t       per = (C + nth - 1) / nth, c0 = ith * per, c1 = std::min(C, c0 + per);
    std::vector<float>  wr(k);
    for (int64_t c = c0; c < c1; ++c) {
        row_f32(w, c * w->ne[1], wr.data(), k);  // w [k, 1, C]: row c
        const float * xr = (const float *) ((const char *) x->data + c * x->nb[1]);
        float *       yr = (float *) ((char *) dst->data + c * dst->nb[1]);
        const float   w0 = wr[0];
        for (int64_t t = 0; t < To; ++t) {
            yr[t] = w0 * xr[t];
        }
        for (int64_t j = 1; j < k; ++j) {
            const float   wj = wr[j];
            const float * xs = xr + j * d;
            for (int64_t t = 0; t < To; ++t) {
                yr[t] += wj * xs[t];
            }
        }
    }
}
