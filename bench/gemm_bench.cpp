/* gemm_bench.cpp -- time a plugin's gemm_nt row on the card, from outside the engine.
 *
 * Loads the built .so with dlopen, builds the RadArgs the engine would build, and times the launch
 * with HIP events. This measures the shipped kernel, not a copy of it.
 *
 *   hipcc -std=c++20 -O2 -I<radiance>/abi gemm_bench.cpp -o gemm_bench -ldl
 *   ./gemm_bench ./libgfx9probe.so            # default M sweep with N=96 K=5120
 *   ./gemm_bench ./libgfx9probe.so 512 96 2048
 */
#include <hip/hip_runtime.h>

#include "rad_abi.h"
#include "rad_abi_layout.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

typedef int (*LaunchFn)(const RadArgs*, RadStream);

static void die(const char* m) { std::fprintf(stderr, "gemm_bench: %s\n", m); std::exit(2); }

int main(int argc, char** argv) {
    const char* so = argc > 1 ? argv[1] : "./libgfx9probe.so";
    const int Mfixed = argc > 2 ? std::atoi(argv[2]) : 0;
    const int N = argc > 3 ? std::atoi(argv[3]) : 96;
    const int K = argc > 4 ? std::atoi(argv[4]) : 5120;

    /* TRAP: initialise HIP before loading the plugin. dlopen'ing a kernel library first leaves the
     * runtime in a state where the first hipMemcpy segfaults inside libamdhip64.so, and the
     * backtrace blames the memcpy rather than the load order. */
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) die("no HIP device");
    std::printf("hip: %d device(s)\n", ndev);

    void* h = dlopen(so, RTLD_NOW | RTLD_LOCAL);
    if (!h) die(dlerror());

    /* ABI check, exactly as the loader does it. */
    auto ver = (uint32_t (*)())dlsym(h, "rad_plugin_abi_version");
    auto lay = (uint64_t (*)())dlsym(h, "rad_plugin_abi_layout");
    auto cnt = (int (*)())dlsym(h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
    if (!ver || !lay || !cnt || !at) die("missing exports");
    if (ver() != RAD_ABI_VERSION) die("ABI version mismatch");
    if (lay() != rad_abi_layout()) die("ABI layout mismatch");

    const RadKernelInfo* row = nullptr;
    for (int i = 0, n = cnt(); i < n; ++i) {
        const RadKernelInfo* r = at(i);
        if (r && r->op && !std::strcmp(r->op, "gemm_nt")) { row = r; break; }
    }
    if (!row || !row->launch) die("no gemm_nt row with a launch");
    LaunchFn launch = (LaunchFn)row->launch;
    std::printf("row %s  op %s  domain %d  dtypes %s\n", row->name, row->op, row->domain, row->dtypes);

    std::vector<int> Ms;
    if (Mfixed) Ms.push_back(Mfixed);
    else for (int m : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}) Ms.push_back(m);

    std::printf("%6s %10s %12s %12s %12s\n", "M", "ms", "TFLOP/s", "GFLOP", "rel");
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> u(-0.05f, 0.05f);

    for (int M : Ms) {
        const size_t na = (size_t)M * K, nb = (size_t)N * K, ny = (size_t)M * N;

        /* bf16 storage, generated once per size. */
                std::vector<uint16_t> ha(na), hb(nb);
        for (auto& v : ha) { float f = u(rng); uint32_t x; std::memcpy(&x, &f, 4);
            v = (uint16_t)((x + 0x7fffu + ((x >> 16) & 1u)) >> 16); }
        for (auto& v : hb) { float f = u(rng); uint32_t x; std::memcpy(&x, &f, 4);
            v = (uint16_t)((x + 0x7fffu + ((x >> 16) & 1u)) >> 16); }

        std::fprintf(stderr,"[M=%d] alloc\n",M); std::fflush(stderr);
        void *da = nullptr, *db = nullptr, *dy = nullptr;
        std::fprintf(stderr,"[M=%d] vecs done\n",M); std::fflush(stderr);
        if (hipMalloc(&da, na * 2) != hipSuccess) die("hipMalloc a");
        if (hipMalloc(&db, nb * 2) != hipSuccess) die("hipMalloc b");
        if (hipMalloc(&dy, ny * 2) != hipSuccess) die("hipMalloc y");
        std::fprintf(stderr,"[M=%d] mallocs ok\n",M); std::fflush(stderr);
        hipMemcpy(da, ha.data(), na * 2, hipMemcpyHostToDevice);
        hipMemcpy(db, hb.data(), nb * 2, hipMemcpyHostToDevice);
        hipMemset(dy, 0, ny * 2);

        std::fprintf(stderr,"[M=%d] copies ok\n",M); std::fflush(stderr);
        RadTensor t[3];
        std::memset(t, 0, sizeof t);
        t[0].data = da; t[0].dtype = RAD_BF16; t[0].rank = 2; t[0].shape[0] = M; t[0].shape[1] = K;
        t[0].stride[1] = 1; t[0].stride[0] = K;
        t[1].data = db; t[1].dtype = RAD_BF16; t[1].rank = 2; t[1].shape[0] = N; t[1].shape[1] = K;
        t[1].stride[1] = 1; t[1].stride[0] = K;
        t[2].data = dy; t[2].dtype = RAD_BF16; t[2].rank = 2; t[2].shape[0] = M; t[2].shape[1] = N;
        t[2].stride[1] = 1; t[2].stride[0] = N;

        RadParam p[4];
        std::memset(p, 0, sizeof p);
        p[0].key = "M"; p[0].kind = RAD_P_RANGE; p[0].ival = M; p[0].ihi = M;
        p[1].key = "N"; p[1].kind = RAD_P_INT; p[1].ival = N;
        p[2].key = "K"; p[2].kind = RAD_P_INT; p[2].ival = K;
        p[3].key = "dtype"; p[3].kind = RAD_P_STR; p[3].sval = "bf16";

        RadArgs a;
        std::memset(&a, 0, sizeof a);
        a.t = t; a.n_t = 3; a.p = p; a.n_p = 4; a.rank = 0; a.world_size = 1;
        a.kv_used_max_h = -1;

        std::fprintf(stderr,"[M=%d] launch\n",M); std::fflush(stderr);
        /* Warm up, then time. */
        int rc = launch(&a, nullptr);
        if (rc != RAD_OK) { std::printf("%6d  launch refused rc=%d\n", M, rc); continue; }
        hipDeviceSynchronize();

        const int iters = M >= 128 ? 200 : 2000;
        hipEvent_t e0, e1;
        hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0, nullptr);
        for (int i = 0; i < iters; ++i) launch(&a, nullptr);
        hipEventRecord(e1, nullptr);
        hipEventSynchronize(e1);
        float ms = 0.f;
        hipEventElapsedTime(&ms, e0, e1);
        ms /= (float)iters;

        const double gflop = 2.0 * (double)M * N * K / 1e9;
        const double tflops = gflop / (ms / 1000.0) / 1000.0;

        /* Pull one row back to show it is not a no-op. */
        std::vector<uint16_t> hy(ny);
        hipMemcpy(hy.data(), dy, ny * 2, hipMemcpyDeviceToHost);
        double acc = 0; for (int i = 0; i < N; ++i) { uint32_t x = (uint32_t)hy[i] << 16; float f;
            std::memcpy(&f, &x, 4); acc += f; }

        std::printf("%6d %10.4f %12.2f %12.3f %12.4f\n", M, ms, tflops, gflop, acc);

        hipFree(da); hipFree(db); hipFree(dy);
        hipEventDestroy(e0); hipEventDestroy(e1);
    }
    return 0;
}
