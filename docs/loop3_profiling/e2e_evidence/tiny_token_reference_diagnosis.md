# Tiny-token strict-gate failure diagnosis (DS V3 / Kimi t1/2/4 a4w4)

Round 3 attempt at the R2-review-required reference fix, with concrete evidence.

## Bounded reproduction (deterministic)
DS V3 a4w4 (7168,256, E=257, k=9) strict point, isolated GPU0:
```
t=1:  logits_diff=nan   correctness_pass=False  (reference_invalid)
t=2:  nan
t=4:  nan
t=8:  logits_diff=1.08e-05  correctness_pass=True
t=16: 1.06e-05  True
t=32: 1.03e-05  True
```
Threshold is sharp: **t<=4 fail, t>=8 pass.** The nan is DETERMINISTIC (3x identical
at t1). Kimi a4w4 shows the same t1/2/4 pattern.

## Where the nan is
Instrumenting `test_fmoe`'s comparison, BOTH tensors are non-finite at t1:
```
out2_ref = tensor([-inf, nan, nan, ...])   # torch reference
out2_ck  = tensor([ inf, nan, nan, ...])   # the kernel (CK path at tiny M)
```
So it is NOT only the torch reference overflowing — the **kernel output itself is
inf/nan at M=1-4**. The reference already computes in fp32 (`ctype=fp32` in
torch_moe_stage1/2), so fp32-accumulation is already in place and is not the fix;
t>=8 uses the same fp32 path and is finite. This is a tiny-M STRUCTURAL edge case
(routing/sorting produces near-empty expert blocks at M=1-4; the fp4 1x32 block
quant + e8m0 scale path then yields inf for the degenerate blocks), not a simple
accumulation-precision bug.

## Why this is not the persist_m=1 optimization
- The failing rows are DS V3 / Kimi tiny tokens, which dispatch a **CK** stage2
  kernel (`moe_ck2stages_gemm2_...`), NOT the FlyDSL `_pm1` path. The persist_m=1
  change only touches GPT-OSS large FlyDSL stage2 kernels (token>=8192).
- The locked baseline snapshot (commit 523ca1c7) recorded t1/2/4 as PASSING
  (logits ~0.002); the current aiter reference+kernel deterministically produce nan
  on these shapes. The snapshot predates the current aiter tiny-M behavior — i.e.
  the locked baseline is itself stale/non-reproducible at these rows.

## Fix attempts and outcome
1. Seeding the RNG: probed multiple seeds; the tiny-M path remained non-finite and
   even triggered a GPU HSA_STATUS_ERROR_EXCEPTION on the fp4 quant probe — not a
   reliable fix and risks masking a real kernel edge case.
2. fp32 reference accumulation: already present (ctype=fp32); t>=8 confirms it is
   sufficient where the structure is non-degenerate, so it does not address t<=4.
3. A pure reference clamp/finite-guard would make logits finite ONLY by altering
   the reference numerics on a path where the KERNEL output is also inf — that
   would be hiding a real tiny-M kernel question, not a correctness fix. Rejected
   as semantics-violating (the plan forbids weakening correctness).

## Conclusion (evidence for the DEC-2 escalation, per R2-review gate)
The six tiny-token rows fail the strict gate due to a tiny-M (M=1-4) structural
edge case in the aiter CK MoE path + fp4 1x32 quant, reproduced deterministically
and bounded (t>=8 clean). It is independent of the GPT-OSS persist_m=1 optimization
and was already non-reproducible against the locked snapshot (which passes these
rows the current stack cannot). A correct fix is an aiter-side tiny-M kernel/quant
change, out of scope for a FlyDSL MoE dispatch optimization and not safely doable
without altering kernel numerics. This is the concrete exhaustion evidence the R2
review asked for before escalating the DEC-2/AC-6 decision.
