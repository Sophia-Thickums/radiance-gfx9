# Implementing the rest — a guide for whoever picks this up

Six device rows work: `add`, `silu_mul`, `rmsnorm`, `gemm_nt`, `cast`, `gather_rows`. A bf16 dense
decoder also needs **`rope`**, **`attn_paged`**, **`kv_store`**, **`logits_gemm`**. This file is what
you need to write them, including the constraints that are not obvious from the headers.

## The constraint that shapes every kernel

`libref` is the oracle, and its contract is **f32 accumulation, IEEE round-to-nearest-even, no FMA and
no reassociation.** That last word is the one that bites: *no reassociation* means you may not
re-order a sum. So:

- **Parallelism comes only from independent output elements**, never from splitting a sum across
  lanes. No lane-tree reductions.
- **Online / flash softmax is not compatible** with a sequential reference, because its running
  rescale `acc *= exp(m_old - m_new)` introduces roundings the reference does not have. Use it only if
  `libref` is itself online with a fixed tile size, and then mirror that tile size exactly.
- The `gemm_nt` row here is bit-exact (`rel_l2 0.00e+00`) because its k-loop walks ascending, the same
  order as the reference. That property is worth preserving where you can.

Read `libref`'s loop nest before writing anything: the order of `d` in a dot product, where `scale`
is applied, the token order of the exp-sum, and whether it normalises `p/l` before P·V or `acc/l`
after. Each of those is a place to disagree with the oracle.

## The hardware, as measured on this machine

gfx900 (Vega 10), 16 GB HBM2, ~484 GB/s, 64 CUs.

- **wave64 only.** Verified by the compiler: `ballot_w32` and `permlanex16` fail with
  `needs target feature ...`. Everything gfx10+ is out.
- **No matrix cores** (`wmma-128b-insts` missing). All FLOPs are VALU.
- **No `v_dot2_f32_f16`** (`needs target feature dot10-insts`). The packed-fp16 path is
  `v_pk_fma_f16` with **fp16** accumulators only — see MEASUREMENTS.md §10, where it measured slower
  *and* less accurate on `gemm_nt`.
- **No bf16 arithmetic or conversion instruction.** `__bf16 -> float` is `v_lshlrev_b32 16`, exact.
- 256 VGPRs a lane, 64 KB LDS a workgroup.

## `attn_paged` — the design, worked

Parameters: `q_len` `head_dim` `gqa` `block_size` `causal` `window` `q_dtype` `kv_dtype` `scale`
`max_ctx` `n_head` `max_seqs`. Operands: `q`, `kv_cache`, `block_table`, `seqused` → `out`. The cache
is paged: `block_table` maps a logical block to a physical one, `block_size` tokens each, and
`seqused` is the live token count per sequence.

**Decode (`q_len = 1`), the case that matters most.**

- **Grid** `(n_kv_heads, num_seqs)`; **block** 256 threads (4 waves). One workgroup owns all `gqa`
  query heads of one KV head, so every K/V byte is loaded once and used `gqa` times.
- Stage the `gqa` query rows into LDS as f32. bf16→f32 is the exact shift.
- Walk the context in tiles of 32 tokens (two pages at `block_size=16`). For a tile, resolve
  `phys = block_table[seq][t / block_size]`, `slot = t % block_size`, and copy that tile's K rows
  into LDS along `head_dim` so global reads coalesce. Pad each row to break bank conflicts.
- Thread `(t = tid % T, h = tid / T)` computes `s = Σ_d q_h[d]·k_t[d]` **serially in `d`**, applying
  `scale` where `libref` applies it. Masked or out-of-range tokens (`t >= seqused`, outside
  `window`) get `-inf`, or are skipped, exactly as the reference does.
- Write the scores to a **global f32 workspace** `[seq][q_head][max_ctx]`. It costs 24 B a token
  against 1 KB of K/V, stays L2-hot, and removes any LDS ceiling on context length.
- Barrier. `m_h = max_t s` is exact in any order, so reduce it in parallel. `l_h = Σ exp(s - m)` is
  **order-sensitive** — sum it serially per head in the reference's token order. Then P·V with the
  same rule: each output element owned by one thread, accumulated serially.

**Prefill (`q_len > 1`, causal)** is the same walk with a triangular mask; the score workspace
becomes `[seq][q_head][q_len][ctx]`, which is why most implementations tile it.

**The three places it goes wrong quietly**: a normalisation applied on the wrong side of P·V; `exp`
differing from the reference's by one ulp and compounding over thousands of tokens; and the `window`
boundary off by one. All three produce plausible numbers, not crashes. `rad-kbench`'s relative-L2
against `libref` is what catches them — which is why every row here is verified before it is called
done.

## The remaining rows, in the order I would do them

1. **`kv_store`** — writes `k`, `v` into the paged cache at `slot_mapping`. Watch: the cache may be
   i8 (codes plus a scale), fp8 (`stored / k_scale`), or a wider dtype; `kv_view()` in `libref` gives
   the layout. Refuse what you do not decode rather than guessing.
2. **`logits_gemm`** — `x @ lm_head^T`. The same maths as `gemm_nt` with `n_vocab` as N; here that is
   248320, so the tiled path does not apply and it needs a wide-N variant.
3. **`rope`** — 47 fixture cases across modes (`norm`, `neox`, `imrope`, yarn scaling, ladders). The
   most surface area for the least speed. Do it late, or leave it on the host.
4. **`attn_paged`** — last, largest, and the one that decides whether the library can serve a model
   at all.

## Traps that cost real time here

- **`--kernels` restricts rather than adds.** Name `libref` too, or every row errors for a missing
  schema.
- **A row no fixture case reaches is reported as a row, not a failure.** `cast` was declared with a
  constraint on `dtype` — a parameter it does not have — so it was never selected and the run still
  passed 93 of 93. Read the `run` column, never only the pass count.
- **Template commas inside `hipLaunchKernelGGL`** split the macro's arguments. Wrap the kernel name
  in parentheses: `hipLaunchKernelGGL((k<T, U>), ...)`.
- **An index dtype cast, not templated**, reads a wider type out of the buffer and faults the queue
  with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. A kernel that claims an encoding it does not
  decode is worse than one that refuses — return `RAD_E_UNSUPPORTED` instead.
- **`dlopen` before the HIP runtime is initialised** segfaults inside `libamdhip64` with a backtrace
  that blames `hipMemcpy`. Initialise first, and even then treat a standalone harness as scratch.
- **Do not commit the linker's intermediate `libfoo.so.0.*` objects**; add them to `.gitignore`.

## What "done" looks like here

Not "fast". The arithmetic in MEASUREMENTS.md §9 says a hand-written `gemm_nt` cannot beat
llama.cpp's Vulkan on this card, because the bar sits at 91% of the f32 peak. **The goal is that a
model runs, correctly, on a Vega — and that the person with the card can build it, verify it and tune
it themselves.** Everything above is arranged for that.
