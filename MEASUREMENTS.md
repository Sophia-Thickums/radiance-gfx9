# Measurements

Everything here was run on one machine on 2026-11-23. Nothing is estimated except where it says so.

## The machine and the stack

| | |
|---|---|
| card under test | AMD Radeon Instinct MI25, `gfx900`, 16 GB HBM2 |
| card present but not used | Vega Frontier Edition, `gfx900`, 16 GB HBM2 |
| engine | radiance, commit `5a80ee2f8f46b429439f3dd224666186a5eb724c` (2026-11-22) |
| ROCm | 10.1.0, HIP 7.16.0-4f6d503 |
| compiler | AMD clang 24.0.0git |
| kernel | Linux 7.3.5-2 |
| test tool | `rad-kbench`, replaying `kernels.rkb` (724 cases, version 6, recorded 2026-11-14 from `qwen3.8-27b-iq8r-dflash.rad`) |

## 1. What the engine does on a gfx900 card with no gfx900 library

```
$ ROCR_VISIBLE_DEVICES=1 rad-kbench
I device: AQL backend, 1 device(s) [0:gfx900]
W kernel plugin 'libr4d' has no device code for device 0: the fat binary has
  code objects for [gfx1201] and this card runs [gfx900:xnack-, gfx9-generic:xnack-].
  Its 621 device kernel(s) are left out of selection here.
1448 checked, 1448 passed, 0 failed, 0 skipped, 0 diverged, 0 capped. worst rel_l2 = 3.442e-03
```

The engine starts, enumerates the card, and states the family target the card also accepts
(`gfx9-generic`). The RDNA4 library is *left out, not failed*, exactly as `docs/PLUGIN.md` §10
says. Everything then runs through `libavx` and `libref` — that is, on the CPU.

## 2. What the RDNA4 kernel source does when aimed at gfx900

Not a recompile, not a flag. The compiler refuses, by name:

```
$ hipcc -x hip --offload-arch=gfx900 -c r4d_attn_paged_window.hip
error: '__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12'
       needs target feature wmma-128b-insts,wavefrontsize32
error: '__builtin_amdgcn_ballot_w32' needs target feature wavefrontsize32
```

and, in the speculative-decoding library:

```
$ cmake --build build -j          # targets gfx1201;gfx900;gfx1036
FAILED: plugin/spechub/CMakeFiles/spechub.dir/spechub.hip.o
plugin/spechub/spechub.hip:626: error:
  '__builtin_amdgcn_permlanex16' needs target feature gfx10-insts
```

That second one means the engine does not fully build out of the box on a machine that contains a
gfx900 card, because `spechub` is compiled for every detected card. Configuring with
`-DRAD_GPU_TARGETS=gfx1201` builds it cleanly. This is a build-configuration fact, not a defect in
the probe.

## 3. bf16 on gfx900

```c
__bf16 x; float f = (float)x;
```

```
gfx900:   v_lshlrev_b32_e32 v1, 16, v1
          v_add_f32_e32 v1, v1, v1
gfx1201:  v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s0
```

Vega has no bf16 ALU and no bf16 conversion instruction. Widening is an integer shift; the
arithmetic is f32. The probe therefore widens by shift, adds in f32, and rounds back with
round-to-nearest-ties-to-even, which is the contract `abi/rad_plugin.h` states for `libref`.

## 4. The probe library

One row, `add`, `bf16`/`f32`, contiguous, built with:

```sh
hipcc -x hip --offload-arch=gfx900 -std=c++20 -O2 -fPIC -shared \
      -I<radiance>/abi -o libgfx9probe.so gfx9probe.hip
```

Code objects in the result:

```
$ llvm-objdump --offloading libgfx9probe.so
amdgcn-amd-amdhsa--gfx900
```

One architecture, the right one.

## 5. The run

```sh
ROCR_VISIBLE_DEVICES=1 rad-kbench --kernels gfx9probe:libref:libavx --report report.md
```

```
libraries    gfx9probe@0.1.0 libref@0.1.0 libavx@0.1.0 (under test: gfx9probe libref libavx)

1458 checked, 1458 passed, 0 failed, 0 skipped, 0 diverged, 0 capped. worst rel_l2 = 3.442e-03
memory: 10 case(s) ran inside guarded allocations, 0 wrote outside one

| library     | version | domain | rows | run     | ops       | graph   | cases | pass | fail | worst rel_l2 |
| gfx9probe   | 0.1.0   | device | 1    | 1 of 1  | 1 of 254  | 1 of 40 | 10    | 10   | 0    | 0.00e+00     |
| libavx      | 0.1.0   | host   | 242  | 40 of 242 | 242 of 254 | 40 of 40 | 724 | 724 | 0    | 3.44e-03   |
| libref      | 0.1.0   | host   | 254  | 40 of 254 | 254 of 254 | 40 of 40 | 724 | 724 | 0    | 3.44e-03   |

time: launch 7.77 s (gfx9probe 0.04 s, libavx 7.73 s)
```

Per case, all ten bf16 `add` cases, `n = 5120`, `M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}`:

```
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=1`   | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=2`   | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=4`   | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=8`   | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=16`  | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=32`  | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=64`  | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=128` | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=256` | ok | 0.00e+00 |
| `add` | `gfx9_add` | gfx9probe | device | `n=5120 dtype=bf16 M=512` | ok | 0.00e+00 |
```

`libavx` on the same cases reports `6.00e-03` — so the device row is not merely within tolerance, it
is closer to the reference than the engine's own host library.

## 6. Two traps worth writing down

- **`--kernels` restricts, it does not add.** Naming only the new library leaves `libref` unloaded,
  and the loader then fails with *"implements op 'add', which no loaded plugin declares a schema
  for"* — an error, not a skip. Name `libref` explicitly.
- **A row that no case reaches is reported as a row, not as a failure.** The first attempt declared
  `dtype IN "f32"`, and every `add` case in the fixture is bf16, so the row was never selected and
  the run still passed. A passing benchmark that never ran your kernel proves nothing; read the
  `run` column, not only the pass count.

## 7. The same library on the Vega Frontier Edition

Both gfx900 cards in this machine were run, not just the Instinct. The library is built for the
architecture (`gfx900`), so it should serve both — but that is a claim, so it was measured.

```
ROCR_VISIBLE_DEVICES=2 rad-kbench --kernels gfx9probe:libref:libavx
I device: AQL backend, 1 device(s) [0:gfx900]

| `gfx9probe` | 0.2.0 | device | 3 | 3 of 3 | 3 of 254 | 3 of 40 | 57 | 57 | 0 | 2.40e-04 |
```

Identical to the Instinct MI25 row: **3 of 3 rows ran, 57 of 57 cases passed, 0 failed**.

```
card                              rows  cases  pass  fail  worst rel_l2
Instinct MI25      (09:00.0)      3/3     57    57     0    2.40e-04
Vega Frontier Ed.  (11:00.0)      3/3     57    57     0    2.40e-04
```

Device order on this machine, for anyone repeating it: `0` gfx1201 (RX 9070 XT), `1` gfx900
(Instinct MI25), `2` gfx900 (Vega Frontier Edition), `3` gfx1036 (iGPU). The Vega runs the desktop,
so it carries ~800 MiB of display memory during the run; that did not change the result.
