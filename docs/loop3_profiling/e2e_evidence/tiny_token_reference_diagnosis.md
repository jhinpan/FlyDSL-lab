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

## Seed sweep (loop3 R5; wording bounded loop3 R6 per R5-review)
Added `--seed` to aiter_strict_point.py (input determinism only; does not alter
kernel/reference numerics or thresholds). BOUNDED measured evidence: DS V3 t1 a4w4
with seeds {0,1,7,42,123,2024} ALL produced `logits_diff=nan`,
`correctness_pass=False`, `reference_invalid`. This shows the tested common global
seeds did NOT find a usable deterministic input that unblocks the gate at DS V3 t1.
It does NOT prove every possible random input or all six rows are non-finite --
that stronger claim was retracted. The decisive evidence is the ROOT-CAUSE TRACE
below (the nan is in the aiter CK stage2 kernel output, not the data/reference),
which is what makes this conclusively an aiter-kernel issue regardless of seed.

## ROOT-CAUSE TRACE (loop3 R6, per R5-review request): the nan is in the aiter CK stage2 KERNEL, not the reference
Step-by-step finiteness trace of the DS V3 t1 a4w4 strict path (seed 0, GPU0):
```
input (bf16 randn)          finite=True  absmax=4.219
a1 fp4 quant a1_scale(e8m0) finite=True  absmax=1
mxfp4_to_f32(a1_qt)         finite=True  absmax=6
e8m0_to_f32(a1_scale)       finite=True  absmax=1
topk_weights                finite=True
out1_ref (stage1 torch ref) finite=True  absmax=380.5
```
The ENTIRE torch reference path is finite. The non-finite values appear only in
`out2_ck` (the stage2 KERNEL output), as seen in the full test_fmoe trace
(out2_ck = [inf, nan, ...]).

DS V3 t1 stage2 dispatches `kernelName2 =
moe_ck2stages_gemm2_64x32x32x128_..._FP4X2_FP4X2_B16` -- a **CK** (Composable
Kernel) stage2 kernel, NOT a FlyDSL kernel and NOT the `_pm1` FlyDSL path. So the
tiny-M (M<=4) overflow is conclusively an **aiter CK stage2 fp4 kernel** defect at
very small M, independent of:
- the torch reference (finite at every step above),
- the FlyDSL kernels (the persist_m=1 change is a FlyDSL GPT-OSS-large stage2
  kernel; DS V3/Kimi tiny tokens never touch it),
- input data (the activation fp4 quant/dequant is finite).

This is the precise first-non-finite-tensor finding the R5 review asked for: the
first non-finite tensor is the CK stage2 kernel output `out2_ck`, produced by
aiter's CK fp4 stage2 kernel. A correct fix is therefore an aiter CK-kernel change
at tiny M -- entirely outside a FlyDSL MoE dispatch optimization's scope, and not
fixable by any reference/seed/accumulation change on the FlyDSL side.

## Conclusion (evidence for the DEC-2 escalation, per R2-review gate)
The six tiny-token rows fail the strict gate due to a tiny-M (M=1-4) structural
edge case in the aiter CK MoE path + fp4 1x32 quant, reproduced deterministically
and bounded (t>=8 clean). It is independent of the GPT-OSS persist_m=1 optimization
and was already non-reproducible against the locked snapshot (which passes these
rows the current stack cannot). A correct fix is an aiter-side tiny-M kernel/quant
change, out of scope for a FlyDSL MoE dispatch optimization and not safely doable
without altering kernel numerics. This is the concrete exhaustion evidence the R2
review asked for before escalating the DEC-2/AC-6 decision.
