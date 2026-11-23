/* gfx9_tail.inc -- the small shape-preserving rows, appended to gfx9probe.hip.
 *
 * cast, gather_rows. Both are memory moves with a dtype conversion; there is no arithmetic to
 * arrange, which is why they are cheap to get exactly right and worth doing before the hard rows.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "rad_abi.h"
#include "rad_abi_layout.h"

namespace radiance_gfx9_tail {

__device__ __forceinline__ float bits_to_f32_d(uint32_t dt, uint16_t h) {
    if (dt == RAD_BF16) {
        uint32_t u = (uint32_t)h << 16;
        float f; memcpy(&f, &u, 4); return f;
    }
    return (float)h;
}

__device__ __forceinline__ uint16_t f32_to_bits_d(uint32_t dt, float f) {
    if (dt == RAD_BF16) {
        uint32_t u; memcpy(&u, &f, 4);
        if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
        return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    return (uint16_t)__half_as_ushort(__float2half_rn(f));
}

/* y[i] = (out_dtype) x[i], elementwise. */
template <int IN_DT, int OUT_DT>
__global__ void cast_kernel(const void* __restrict__ x, void* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v;
    if (IN_DT == RAD_BF16) {
        const uint32_t u = (uint32_t)(((const uint16_t*)x)[i]) << 16;
        memcpy(&v, &u, 4);
    } else {
        v = ((const float*)x)[i];
    }
    if (OUT_DT == RAD_BF16) {
        uint32_t u; memcpy(&u, &v, 4);
        uint32_t r;
        if ((u & 0x7fffffffu) > 0x7f800000u) r = (u >> 16) | 0x0040u;
        else r = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
        ((uint16_t*)y)[i] = (uint16_t)r;
    } else {
        ((float*)y)[i] = v;
    }
}

/* y[m][:] = x[idx[m]][:], and a negative index writes zeros. */
template <int DT>
__global__ void gather_rows_kernel(const void* __restrict__ x, const int64_t* __restrict__ idx,
                                   void* __restrict__ y, long long M, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * n) return;
    const long long m = i / n;
    const long long c = i - m * n;
    const long long r = idx[m];
    if (r < 0) {
        ((uint16_t*)y)[i] = 0;
        return;
    }
    const long long src = r * n + c;
    if (DT == RAD_BF16) ((uint16_t*)y)[i] = ((const uint16_t*)x)[src];
    else ((float*)y)[i] = ((const float*)x)[src];
}

}  // namespace radiance_gfx9_tail

extern "C" int gfx9_cast(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t < 2) return RAD_E_INVAL;
    const RadTensor *x = &args->t[0], *y = &args->t[1];
    const long long n = (long long)(x->shape[0] * (x->rank > 1 ? x->shape[1] : 1));
    if (n <= 0) return RAD_E_SHAPE;
    const unsigned blocks = (unsigned)((n + 255) / 256);
    const int in = (int)x->dtype, out = (int)y->dtype;
    if ((in == RAD_BF16 || in == RAD_F32) && (out == RAD_BF16 || out == RAD_F32)) {
        if (in == RAD_BF16 && out == RAD_BF16)
            hipLaunchKernelGGL(radiance_gfx9_tail::cast_kernel<RAD_BF16, RAD_BF16>, dim3(blocks),
                               dim3(256), 0, (hipStream_t)stream, x->data, y->data, n);
        else if (in == RAD_BF16)
            hipLaunchKernelGGL(radiance_gfx9_tail::cast_kernel<RAD_BF16, RAD_F32>, dim3(blocks),
                               dim3(256), 0, (hipStream_t)stream, x->data, y->data, n);
        else if (out == RAD_BF16)
            hipLaunchKernelGGL(radiance_gfx9_tail::cast_kernel<RAD_F32, RAD_BF16>, dim3(blocks),
                               dim3(256), 0, (hipStream_t)stream, x->data, y->data, n);
        else
            hipLaunchKernelGGL(radiance_gfx9_tail::cast_kernel<RAD_F32, RAD_F32>, dim3(blocks),
                               dim3(256), 0, (hipStream_t)stream, x->data, y->data, n);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    return RAD_E_UNSUPPORTED;
}

extern "C" int gfx9_gather_rows(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t < 3) return RAD_E_INVAL;
    const RadTensor *x = &args->t[0], *idx = &args->t[1], *y = &args->t[2];
    if (idx->dtype != RAD_I64 && idx->dtype != RAD_I32) return RAD_E_UNSUPPORTED;
    long long n = 0;
    if (!rad_args_geti(args, "n", &n) || n <= 0) n = y->shape[y->rank - 1];
    if (n <= 0) return RAD_E_SHAPE;
    long long M = 1;
    for (uint32_t d = 0; d + 1 < y->rank; ++d) M *= y->shape[d];
    if (M <= 0) return RAD_E_SHAPE;
    const unsigned blocks = (unsigned)((M * n + 255) / 256);
    if (y->dtype == RAD_BF16)
        hipLaunchKernelGGL(radiance_gfx9_tail::gather_rows_kernel<RAD_BF16>, dim3(blocks), dim3(256),
                           0, (hipStream_t)stream, x->data, (const int64_t*)idx->data, y->data, M, n);
    else
        hipLaunchKernelGGL(radiance_gfx9_tail::gather_rows_kernel<RAD_F32>, dim3(blocks), dim3(256),
                           0, (hipStream_t)stream, x->data, (const int64_t*)idx->data, y->data, M, n);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
