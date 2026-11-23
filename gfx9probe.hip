/* gfx9probe.hip -- a device kernel library for AMD gfx900 (Vega 10).
 *
 * Rows so far: add, rmsnorm, silu_mul. More as they are written and verified.
 *
 * WHY: every device library bundled with radiance targets RDNA4, RDNA3 or Strix Halo. Aiming that
 * source at gfx900 fails by name -- no matrix cores (wmma-128b-insts), no wave32. This file is the
 * other route: kernels written for what Vega 10 actually has. Wave64 only, no bf16 ALU (so bf16 is
 * widened by a shift and rounded back with round-to-nearest-even), 256 VGPRs, 64 KB LDS.
 *
 * Every row is checked numerically by rad-kbench against libref on the card. A row that no fixture
 * case reaches is reported as a row, not a failure -- read the `run` column, not only the passes.
 *
 * Built out of tree against the abi/ headers alone, as docs/PLUGIN.md says a third party would.
 */
#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_abi_layout.h"
#include "rad_plugin.h"

#include <cstring>

namespace {

constexpr int kThreads = 256;

/* ------------------------------------------------------------------ the two dtypes
 * bf16 has no arithmetic on this chip at all, so every bf16 op is: shift left 16, compute in f32,
 * round back with IEEE round-to-nearest-ties-to-even. That is libref's contract, and it is why the
 * device rows agree with the oracle to the last bit rather than to a tolerance.
 */
__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

template <int DT>
__device__ __forceinline__ float ld(const void* p, long long i);
template <int DT>
__device__ __forceinline__ void st(void* p, long long i, float v);

template <>
__device__ __forceinline__ float ld<RAD_F32>(const void* p, long long i) {
    return ((const float*)p)[i];
}
template <>
__device__ __forceinline__ void st<RAD_F32>(void* p, long long i, float v) {
    ((float*)p)[i] = v;
}
template <>
__device__ __forceinline__ float ld<RAD_BF16>(const void* p, long long i) {
    return bf16_to_f32(((const uint16_t*)p)[i]);
}
template <>
__device__ __forceinline__ void st<RAD_BF16>(void* p, long long i, float v) {
    ((uint16_t*)p)[i] = f32_to_bf16(v);
}

bool contiguous(const RadTensor* t) {
    if (!t || !t->data) return false;
    int64_t expect = 1;
    for (int d = (int)t->rank - 1; d >= 0; --d) {
        if (t->stride[d] != expect) return false;
        expect *= t->shape[d];
    }
    return true;
}

long long numel(const RadTensor* t) {
    long long n = 1;
    for (uint32_t d = 0; d < t->rank; ++d) n *= t->shape[d];
    return n;
}

float param_f32(const RadArgs* a, const char* key, float dflt) {
    const RadParam* p = rad_param_find(a->p, a->n_p, key);
    if (!p) return dflt;
    if (p->kind == RAD_P_F64) return (float)p->dval;
    if (p->kind == RAD_P_INT) return (float)p->ival;
    return dflt;
}

/* ================================================================== add
 * y = (a + b) / div, with `b` broadcast across rows when its element count is n. libref's rule.
 */
template <int DT>
__global__ void add_kernel(const void* __restrict__ a, const void* __restrict__ b,
                           void* __restrict__ y, long long rows, long long n,
                           long long b_stride_rows, float div) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const long long r = i / n;
    float s = ld<DT>(a, i) + ld<DT>(b, i - r * n + r * b_stride_rows);
    if (div != 1.0f) s = s / div;
    st<DT>(y, i, s);
}

/* ================================================================== silu_mul
 * y[r][i] = silu(gate_up[r][i]) * gate_up[r][n + i].  One row of gate_up is 2n wide.
 */
template <int DT>
__global__ void silu_mul_kernel(const void* __restrict__ gu, void* __restrict__ y,
                                long long rows, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const long long r = i / n;
    const long long c = i - r * n;
    const long long base = r * 2 * n;
    const float g = ld<DT>(gu, base + c);
    const float u = ld<DT>(gu, base + n + c);
    const float s = g / (1.0f + __expf(-g));   /* silu */
    st<DT>(y, i, s * u);
}

/* ================================================================== rmsnorm
 * y = x * rsqrt(mean(x^2) + eps) * (w + wadd) * mult, one workgroup of one wave per row.
 *
 * The sum is taken per lane over a stride-64 walk and then reduced across the wave, so the order
 * differs from libref's sequential ascending accumulation. The fixture's tolerance covers that;
 * a bit-exact rmsnorm would have to be single-lane, which is not worth the time it costs.
 */
template <int DT>
__global__ void rmsnorm_kernel(const void* __restrict__ x, const void* __restrict__ w,
                               void* __restrict__ y, long long rows, long long n,
                               float eps, float wadd, float mult, int has_w) {
    const long long r = blockIdx.x;
    if (r >= rows) return;
    const long long base = r * n;
    const int lane = threadIdx.x;

    float ss = 0.0f;
    for (long long i = lane; i < n; i += blockDim.x) {
        const float v = ld<DT>(x, base + i);
        ss += v * v;
    }
    /* wave64 reduction: 64 lanes, six steps. */
    for (int off = 32; off > 0; off >>= 1) ss += __shfl_down(ss, off);
    __shared__ float s_ss;
    if (lane == 0) s_ss = ss;
    __syncthreads();

    const float s = rsqrtf(s_ss / (float)n + eps);
    for (long long i = lane; i < n; i += blockDim.x) {
        const float v = ld<DT>(x, base + i);
        const float gain = has_w ? (ld<DT>(w, i) + wadd) : 1.0f;
        st<DT>(y, base + i, v * s * gain * mult);
    }
}

/* ================================================================== gemm_nt
 * y[M,N] = a[M,K] @ b[N,K]^T, with b stored N-major and K contiguous (the natural weight layout).
 *
 * First version: correct before fast. One thread owns four consecutive outputs along N, so each
 * `a` element it loads feeds four FMAs and the k-loop carries four f32 accumulators. No LDS
 * staging yet -- that is the optimisation pass, and it will be measured against this rather than
 * replacing it blind.
 */
template <int DT>
__global__ void gemm_nt_kernel(const void* __restrict__ a, const void* __restrict__ b,
                               void* __restrict__ y, int M, int N, int K, int n4) {
    const int row = blockIdx.y;
    const int col4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col4 >= n4) return;
    const int n0 = col4 * 4;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    const long long abase = (long long)row * K;
    const long long b0 = (long long)(n0 + 0) * K;
    const long long b1 = (long long)(n0 + 1) * K;
    const long long b2 = (long long)(n0 + 2) * K;
    const long long b3 = (long long)(n0 + 3) * K;

    const int rem = N - n0;   /* 1..4 live columns in this group */

    for (int k = 0; k < K; ++k) {
        const float av = ld<DT>(a, abase + k);
        acc0 += av * ld<DT>(b, b0 + k);
        if (rem > 1) acc1 += av * ld<DT>(b, b1 + k);
        if (rem > 2) acc2 += av * ld<DT>(b, b2 + k);
        if (rem > 3) acc3 += av * ld<DT>(b, b3 + k);
    }

    const long long ybase = (long long)row * N + n0;
    st<DT>(y, ybase + 0, acc0);
    if (rem > 1) st<DT>(y, ybase + 1, acc1);
    if (rem > 2) st<DT>(y, ybase + 2, acc2);
    if (rem > 3) st<DT>(y, ybase + 3, acc3);
}

/* ================================================================== gemm_nt (tiled)
 * v2. Three things were killing v1: the staging loop divided by a non-constant kk per element
 * (20+ cycles each), every element was a separate 16-bit global load (2000 dependent loads a thread
 * with no overlap, which is why even M=1 cost 4.5 ms), and the b tile was N-major so the four reads
 * in the inner loop all landed in the same LDS bank.
 *
 * Now: the tile is staged with 16-byte vector loads, stored k-major as bf16 so consecutive columns
 * are consecutive addresses, and the inner loop reads one bank per column.
 */
constexpr int kTM = 8;          /* rows per workgroup */
constexpr int kTK = 32;         /* k values staged per step */
constexpr int kThreads2 = 256;
constexpr int kMaxN = 96;       /* the shapes that matter; the launcher refuses larger */
constexpr int kVec = 8;         /* bf16 elements per 16-byte load */

template <int DT>
__global__ void gemm_nt_tiled(const void* __restrict__ a, const void* __restrict__ b,
                              void* __restrict__ y, int M, int N, int K) {
    __shared__ float s_a[kTK][kTM];         /* k-major: consecutive rows are consecutive floats */
    __shared__ float s_b[kTK][kMaxN];

    const int row0 = blockIdx.x * kTM;
    const int rows = min(kTM, M - row0);
    if (rows <= 0) return;

    const int tid = threadIdx.x;
    const int tpr = kThreads2 / kTM;                 /* threads per row */
    const int tr = tid / tpr;
    const int tc = tid - tr * tpr;
    const int per = (N + tpr - 1) / tpr;             /* columns per thread */

    float acc[4];
    for (int i = 0; i < 4; ++i) acc[i] = 0.f;

    constexpr int kStep = 4;        /* elements a thread stages per pass */

    for (int k0 = 0; k0 < K; k0 += kTK) {
        const int kk = min(kTK, K - k0);

        /* b: N rows of kk values, converted once into a k-major tile. */
        for (int i = tid * kStep; i < N * kk; i += kThreads2 * kStep) {
            const int nn = i / kk;
            const int cc = i - nn * kk;
            const long long base = (long long)nn * K + k0 + cc;
            if (cc + kStep <= kk) {
                if (DT == RAD_BF16) {
                    /* one 8-byte load of four bf16 */
                    const uint2 v = *(const uint2*)((const uint16_t*)b + base);
                    const uint16_t* h = (const uint16_t*)&v;
                    for (int q = 0; q < 4; ++q) s_b[cc + q][nn] = bf16_to_f32(h[q]);
                } else {
                    const float4 v = *(const float4*)((const float*)b + base);
                    s_b[cc + 0][nn] = v.x;
                    s_b[cc + 1][nn] = v.y;
                    s_b[cc + 2][nn] = v.z;
                    s_b[cc + 3][nn] = v.w;
                }
            } else {
                for (int q = 0; cc + q < kk; ++q) s_b[cc + q][nn] = ld<DT>(b, base + q);
            }
        }

        /* a: this tile's rows, same layout. */
        for (int i = tid * kStep; i < kTM * kk; i += kThreads2 * kStep) {
            const int rr = i / kk;
            const int cc = i - rr * kk;
            for (int q = 0; q < kStep && cc + q < kk; ++q)
                s_a[cc + q][rr] = (rr < rows) ? ld<DT>(a, (long long)(row0 + rr) * K + k0 + cc + q)
                                               : 0.f;
        }
        __syncthreads();

        if (tr < rows) {
            const int c0 = tc * per;
            const int live = min(4, N - c0);
            for (int cc = 0; cc < kk; ++cc) {
                const float av = s_a[cc][tr];
                const float* bp = &s_b[cc][c0];
                acc[0] += av * bp[0];
                if (live > 1) acc[1] += av * bp[1];
                if (live > 2) acc[2] += av * bp[2];
                if (live > 3) acc[3] += av * bp[3];
            }
        }
        __syncthreads();
    }

    if (tr < rows) {
        const int c0 = tc * per;
        const long long ybase = (long long)(row0 + tr) * N + c0;
        for (int j = 0; j < 4 && c0 + j < N; ++j) st<DT>(y, ybase + j, acc[j]);
    }
}

}  // namespace

/* ================================================================== launchers */

extern "C" int gfx9_add(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t < 3) return RAD_E_INVAL;
    const RadTensor *a = &args->t[0], *b = &args->t[1], *y = &args->t[2];
    const uint32_t dt = y->dtype;
    if (dt != RAD_BF16 && dt != RAD_F32) return RAD_E_DTYPE;
    if (a->dtype != dt || b->dtype != dt) return RAD_E_DTYPE;
    if (!contiguous(a) || !contiguous(b) || !contiguous(y)) return RAD_E_STRIDE;

    long long n = 0;
    if (!rad_args_geti(args, "n", &n) || n <= 0) n = numel(y);
    if (n <= 0) return RAD_E_SHAPE;
    const long long yn = numel(y);
    if (yn % n) return RAD_E_SHAPE;
    const long long rows = yn / n;
    if (numel(a) < yn) return RAD_E_SHAPE;
    const long long bn = numel(b);
    const bool bcast = (bn == n && rows > 1);
    if (!bcast && bn < yn) return RAD_E_SHAPE;
    const long long b_stride_rows = bcast ? 0 : n;

    const float div = param_f32(args, "div", 1.0f);
    if (!(div != 0.0f)) return RAD_E_INVAL;

    const unsigned blocks = (unsigned)((yn + kThreads - 1) / kThreads);
    if (dt == RAD_BF16)
        hipLaunchKernelGGL(add_kernel<RAD_BF16>, dim3(blocks), dim3(kThreads), 0,
                           (hipStream_t)stream, a->data, b->data, y->data, rows, n,
                           b_stride_rows, div);
    else
        hipLaunchKernelGGL(add_kernel<RAD_F32>, dim3(blocks), dim3(kThreads), 0,
                           (hipStream_t)stream, a->data, b->data, y->data, rows, n,
                           b_stride_rows, div);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int gfx9_silu_mul(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t < 2) return RAD_E_INVAL;
    const RadTensor *gu = &args->t[0], *y = &args->t[1];
    const uint32_t dt = y->dtype;
    if (dt != RAD_BF16 && dt != RAD_F32) return RAD_E_DTYPE;
    if (gu->dtype != dt) return RAD_E_DTYPE;
    if (!contiguous(gu) || !contiguous(y)) return RAD_E_STRIDE;

    long long n = 0;
    if (!rad_args_geti(args, "n", &n) || n <= 0) n = numel(y);
    if (n <= 0) return RAD_E_SHAPE;
    const long long yn = numel(y);
    if (yn % n) return RAD_E_SHAPE;
    const long long rows = yn / n;
    if (numel(gu) < rows * 2 * n) return RAD_E_SHAPE;

    const unsigned blocks = (unsigned)((yn + kThreads - 1) / kThreads);
    if (dt == RAD_BF16)
        hipLaunchKernelGGL(silu_mul_kernel<RAD_BF16>, dim3(blocks), dim3(kThreads), 0,
                           (hipStream_t)stream, gu->data, y->data, rows, n);
    else
        hipLaunchKernelGGL(silu_mul_kernel<RAD_F32>, dim3(blocks), dim3(kThreads), 0,
                           (hipStream_t)stream, gu->data, y->data, rows, n);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int gfx9_rmsnorm(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t < 3) return RAD_E_INVAL;
    const RadTensor *x = &args->t[0], *w = &args->t[1], *y = &args->t[2];
    const uint32_t dt = y->dtype;
    if (dt != RAD_BF16 && dt != RAD_F32) return RAD_E_DTYPE;
    if (x->dtype != dt || y->dtype != dt) return RAD_E_DTYPE;
    const int has_w = (w && w->data) ? 1 : 0;
    if (has_w && w->dtype != dt) return RAD_E_DTYPE;
    if (!contiguous(x) || !contiguous(y) || (has_w && !contiguous(w))) return RAD_E_STRIDE;

    long long n = 0;
    if (!rad_args_geti(args, "n", &n) || n <= 0) {
        if (has_w && w->rank) n = w->shape[w->rank - 1];
    }
    if (n <= 0) return RAD_E_SHAPE;
    const long long xn = numel(x);
    if (xn % n) return RAD_E_SHAPE;
    const long long rows = xn / n;
    if (numel(y) < rows * n) return RAD_E_SHAPE;
    if (has_w && numel(w) < n) return RAD_E_SHAPE;

    const float eps = param_f32(args, "eps", 1e-6f);
    const float wadd = param_f32(args, "wadd", 0.0f);
    const float mult = param_f32(args, "mult", 1.0f);
    if (!has_w && wadd != 0.0f) return RAD_E_INVAL;

    if (dt == RAD_BF16)
        hipLaunchKernelGGL(rmsnorm_kernel<RAD_BF16>, dim3((unsigned)rows), dim3(64), 0,
                           (hipStream_t)stream, x->data, has_w ? w->data : nullptr, y->data,
                           rows, n, eps, wadd, mult, has_w);
    else
        hipLaunchKernelGGL(rmsnorm_kernel<RAD_F32>, dim3((unsigned)rows), dim3(64), 0,
                           (hipStream_t)stream, x->data, has_w ? w->data : nullptr, y->data,
                           rows, n, eps, wadd, mult, has_w);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int gfx9_gemm_nt(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t < 3) return RAD_E_INVAL;
    const RadTensor *a = &args->t[0], *b = &args->t[1], *y = &args->t[2];
    const uint32_t dt = y->dtype;
    if (dt != RAD_BF16 && dt != RAD_F32) return RAD_E_DTYPE;
    if (a->dtype != dt || b->dtype != dt) return RAD_E_DTYPE;
    if (!contiguous(a) || !contiguous(b) || !contiguous(y)) return RAD_E_STRIDE;

    long long M = 0, N = 0, K = 0;
    if (!rad_args_geti(args, "M", &M)) M = y->rank ? y->shape[0] : 0;
    if (!rad_args_geti(args, "N", &N) || N <= 0) N = y->rank ? y->shape[y->rank - 1] : 0;
    if (!rad_args_geti(args, "K", &K) || K <= 0) K = b->rank ? b->shape[b->rank - 1] : 0;
    if (M <= 0 || N <= 0 || K <= 0) return RAD_E_SHAPE;
    if (M > 65535) return RAD_E_SHAPE;
    if (numel(a) < M * K) return RAD_E_SHAPE;
    if (numel(b) < N * K) return RAD_E_SHAPE;
    if (numel(y) < M * N) return RAD_E_SHAPE;

    const int n4 = (int)((N + 3) / 4);

    if (N <= kMaxN) {
        const dim3 grid((unsigned)((M + kTM - 1) / kTM), 1, 1);
        if (dt == RAD_BF16)
            hipLaunchKernelGGL(gemm_nt_tiled<RAD_BF16>, grid, dim3(kThreads2), 0,
                               (hipStream_t)stream, a->data, b->data, y->data, (int)M, (int)N, (int)K);
        else
            hipLaunchKernelGGL(gemm_nt_tiled<RAD_F32>, grid, dim3(kThreads2), 0,
                               (hipStream_t)stream, a->data, b->data, y->data, (int)M, (int)N, (int)K);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }

    const int threads = 96;                      /* 24 groups of 4 columns */
    const dim3 grid((unsigned)((n4 + threads - 1) / threads), (unsigned)M, 1);
    if (dt == RAD_BF16)
        hipLaunchKernelGGL(gemm_nt_kernel<RAD_BF16>, grid, dim3(threads), 0, (hipStream_t)stream,
                           a->data, b->data, y->data, (int)M, (int)N, (int)K, n4);
    else
        hipLaunchKernelGGL(gemm_nt_kernel<RAD_F32>, grid, dim3(threads), 0, (hipStream_t)stream,
                           a->data, b->data, y->data, (int)M, (int)N, (int)K, n4);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

namespace {

const RadConstraint kCons[] = { RAD_CIN("dtype", "bf16 f32") };

const RadKernelInfo kRows[] = {
    { "gfx9_add", "add", "elem", "elementwise sum y=(a+b)/div", "any", "bf16 f32",
      RAD_DOMAIN_DEVICE, 0, kCons, 1,
      nullptr, 0, nullptr,
      nullptr, nullptr,
      gfx9_add,
      nullptr, nullptr, nullptr, nullptr },

    { "gfx9_silu_mul", "silu_mul", "elem", "silu-gated product over a fused gate_up", "any",
      "bf16 f32",
      RAD_DOMAIN_DEVICE, 0, kCons, 1,
      nullptr, 0, nullptr,
      nullptr, nullptr,
      gfx9_silu_mul,
      nullptr, nullptr, nullptr, nullptr },

    { "gfx9_rmsnorm", "rmsnorm", "norm", "RMS norm with gain, wadd and mult", "one wave a row",
      "bf16 f32",
      RAD_DOMAIN_DEVICE, 0, kCons, 1,
      nullptr, 0, nullptr,
      nullptr, nullptr,
      gfx9_rmsnorm,
      nullptr, nullptr, nullptr, nullptr },

    { "gfx9_gemm_nt", "gemm_nt", "gemm", "y = a @ b^T, four outputs a thread, f32 accumulate",
      "M any, N any, K any",
      "bf16 f32",
      RAD_DOMAIN_DEVICE, 0, kCons, 1,
      nullptr, 0, nullptr,
      nullptr, nullptr,
      gfx9_gemm_nt,
      nullptr, nullptr, nullptr, nullptr },
};

}  // namespace

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = { RAD_PLUGIN_KERNEL, "gfx9probe", "0.2.0",
                                        "gfx900 (Vega 10) device kernels: add, silu_mul, rmsnorm",
                                        "gfx900" };
    return &info;
}

int rad_kernel_schema_count(void) { return 0; }
const RadOpSchema* rad_kernel_schema_at(int) { return nullptr; }
int rad_kernel_count(void) { return (int)(sizeof kRows / sizeof kRows[0]); }
const RadKernelInfo* rad_kernel_at(int i) {
    return i >= 0 && i < (int)(sizeof kRows / sizeof kRows[0]) ? &kRows[i] : nullptr;
}

}  // extern "C"
