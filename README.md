# radiance-gfx9

A **device kernel library for AMD gfx900** (Vega 10 — Instinct MI25, Vega Frontier Edition, Vega 56/64) for the [radiance](https://codeberg.org/StillDeadcode/radiance) inference engine, and the measurements that prove the engine's plugin path carries a non-RDNA4 card.

## Why this exists

Every device kernel library shipped with radiance targets RDNA4 (`gfx1200`, `gfx1201`), RDNA3 (`gfx11xx`) or Strix Halo (`gfx1151`). Aim the existing RDNA4 kernel source at `gfx900` and the compiler refuses, naming the missing hardware:

```
error: '__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12'
       needs target feature wmma-128b-insts,wavefrontsize32
error: '__builtin_amdgcn_ballot_w32' needs target feature wavefrontsize32
error: '__builtin_amdgcn_permlanex16' needs target feature gfx10-insts
```

Two things Vega 10 does not have: **matrix cores** and **wave32**. That is settled and it is not a build flag.

What was *not* settled — and had never been tested — is whether the engine's kernel-library ABI can carry a device library for a gfx9 card at all. The engine core says it assumes nothing about a card's wave size, LDS, fp8 hardware or links, and `docs/PLUGIN.md` §10 documents writing a library for another card. That is a claim in a document. This repository is the test of it.

## What was measured

The probe is one row: the conventional `add` op (`y = (a + b) / div`), bf16 and f32, contiguous, with `b` broadcast across rows when its element count equals `n` — libref's own rule from `ref_elementwise.cpp`.

Built out of tree with `hipcc --offload-arch=gfx900` against the `abi/` headers alone, and run on an **Instinct MI25** through `rad-kbench`:

```
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=1`   | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=512` | ok | 0.00e+00 |
| `add` | `avx_add`  | libavx    | host   | `n=5120 dtype=bf16 M=512` | ok | 6.00e-03 |
```

**10 of 10 cases passed, worst rel_l2 0.00e+00** — bit-exact against the host reference, and closer than the engine's own AVX-512 host library on the same cases. The row was selected over `libavx` for every one, and it executed in the `device` domain.

So:

- the plugin loads on a gfx900 card;
- the engine selects it in preference to the host library;
- its results are checked, not assumed;
- and it runs on the card.

That is the whole claim. A real gfx900 library is now a question of work, not of possibility.

## Files

| file | what |
|---|---|
| `gfx9probe.hip` | the library: one `add` row, bf16 + f32, plus the bf16 conversions done the way gfx900 requires |
| `build.sh` | builds `libgfx9probe.so` for `gfx900` |
| `MEASUREMENTS.md` | every number, the exact commands, and what was ruled out |

## Build and run

```sh
git clone https://codeberg.org/StillDeadcode/radiance && cd radiance
./build.sh
export RADIANCE_HOME=$PWD/build/radiance_home

./path/to/build.sh                       # this repository
cp libgfx9probe.so "$RADIANCE_HOME/kernels/"

ROCR_VISIBLE_DEVICES=0 rad-kbench --kernels gfx9probe:libref:libavx
```

`--kernels` **restricts** loading to the libraries it names, so name `libref` too: it is what declares the op schemas, and a row whose op has no declared schema is an error, not a skip.

## The bf16 detail, which is the interesting one

gfx900 has no bf16 arithmetic and **no bf16 conversion instruction** — measured, not assumed:

```
gfx900:   v_lshlrev_b32_e32 v1, 16, v1      ; __bf16 -> float is a shift
```

So bf16 here is widened by shift to f32, added in f32, and rounded back with IEEE round-to-nearest-ties-to-even. That is libref's contract, and it is why the comparison is exact. The alternative — narrowing weights to fp16 to keep the packed-fp16 rate — buys throughput at a range cost that has not yet been measured.

## Where this goes next

Ordered by what a probe can actually decide, not by what is exciting:

1. `rmsnorm` and `residual add`, then `gemv` for `M=1` decode. Decode is bandwidth-bound on both engines, so the honest target there is parity, not a win.
2. A tiled fp16 `gemm` for prefill, against a `llama-bench` Vulkan baseline measured on the same card first. If it cannot plausibly reach 1.3× that baseline, the project stops there and this repository becomes a negative result worth having.
3. `attn` prefill and decode. The largest kernel by far, and the last one.

The prefill prize, if the fp16 route holds, is roughly 1.2–2× over llama.cpp's Vulkan path on these cards. If bf16 numerics force fp32 compute, the ceiling halves and it is closer to a wash.

## Licence

Apache-2.0, the same as radiance. The `abi/` headers this builds against are radiance's, © Deadcode and the radiance contributors.
