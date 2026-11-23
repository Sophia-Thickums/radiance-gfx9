/* gfx9probe.hip -- the smallest possible DEVICE kernel library for gfx900 (Vega 10).
 *
 * One row: the conventional `add` op, bf16 and f32, contiguous. y = (a + b) / div, with `b`
 * broadcast across the rows when its element count equals n (libref's rule,
 * ref_elementwise.cpp elementwise2).
 *
 * WHY IT EXISTS: to answer one question with a measurement instead of an inference -- can this
 * engine's kernel-library ABI carry a device library for a card that is NOT RDNA4? Every bundled
 * device library is RDNA4/RDNA3/Strix Halo; nothing in the tree targets gfx9. If this loads, gets
 * selected over the host reference, and passes rad-kbench on the MI25, then the path exists and a
 * real gfx900 library is a question of work rather than possibility.
 *
 * The bf16 case is the interesting one: gfx900 has no bf16 ALU and no bf16 conversion instruction
 * (measured: __bf16 -> float compiles to `v_lshlrev_b32 16`). So bf16 here is widened by shift,
 * added in f32, and rounded back with IEEE round-to-nearest-even, which is libref's contract.
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

__device__ __forceinline__ float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* IEEE round-to-nearest, ties to even -- the same rule as rad_plugin.h's rad_f32_to_bf16. */
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

__global__ void add_kernel_bf16(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b,
                                uint16_t* __restrict__ y, long long rows, long long n,
                                long long b_stride_rows, float div) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const long long r = i / n;
    const float u = bf16_to_f32(a[i]);
    const float v = bf16_to_f32(b[i - r * n + r * b_stride_rows]);
    float s = u + v;
    if (div != 1.0f) s = s / div;
    y[i] = f32_to_bf16(s);
}

__global__ void add_kernel_f32(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ y, long long rows, long long n,
                               long long b_stride_rows, float div) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const long long r = i / n;
    float s = a[i] + b[i - r * n + r * b_stride_rows];
    if (div != 1.0f) s = s / div;
    y[i] = s;
}

/* Contiguous, row-major: what the probe claims and nothing more. */
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

}  // namespace

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

    float div = 1.0f;
    const RadParam* p = rad_param_find(args->p, args->n_p, "div");
    if (p) {
        if (p->kind == RAD_P_F64) div = (float)p->dval;
        else if (p->kind == RAD_P_INT) div = (float)p->ival;
    }
    if (!(div != 0.0f)) return RAD_E_INVAL;

    const unsigned blocks = (unsigned)((yn + kThreads - 1) / kThreads);
    if (dt == RAD_BF16) {
        hipLaunchKernelGGL(add_kernel_bf16, dim3(blocks), dim3(kThreads), 0, (hipStream_t)stream,
                           (const uint16_t*)a->data, (const uint16_t*)b->data,
                           (uint16_t*)y->data, rows, n, b_stride_rows, div);
    } else {
        hipLaunchKernelGGL(add_kernel_f32, dim3(blocks), dim3(kThreads), 0, (hipStream_t)stream,
                           (const float*)a->data, (const float*)b->data, (float*)y->data, rows, n,
                           b_stride_rows, div);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

namespace {

const RadConstraint kCons[] = { RAD_CIN("dtype", "bf16 f32") };

const RadKernelInfo kRows[] = {
    { "gfx9_add", "add", "elem", "elementwise sum y=(a+b)/div, gfx900 probe", "any",
      "bf16 f32",
      RAD_DOMAIN_DEVICE, 0,
      kCons, 1,
      nullptr, 0, nullptr,
      nullptr, nullptr,
      gfx9_add,
      nullptr, nullptr, nullptr, nullptr },
};

}  // namespace

extern "C" {

uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT

const RadPluginInfo* rad_plugin_info(void) {
    static const RadPluginInfo info = { RAD_PLUGIN_KERNEL, "gfx9probe", "0.1.0",
                                        "gfx900 (Vega 10) device kernel library, one row",
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
