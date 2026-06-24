# a8w4 Strict-Path Correctness Evidence (locked ref 523ca1c7)

All a8w4 (fp8 activation x fp4 weight) points run through the strict, model-correct
aiter path (`scripts/aiter_strict_point.py`: true per-model activation/gate,
`strict_accuracy=True`). Correctness is gated on `logits_diff <= 0.01`. a8w4 is
correctness-BLOCKED in this environment (see the `kernels/moe_tuning_spec.py` quarantine
note). Categories: `correctness` = strict accuracy assertion (logits ~0.98);
`runtime` = kernel/runtime rejection (e.g. Unsupported scales/output); `pass` = logits<=0.01.

| model | total | correctness-fail | runtime-fail | pass |
|---|---|---|---|---|
| deepseek_v3 | 16 | 4 | 12 | 0 |
| deepseek_v4 | 16 | 10 | 6 | 0 |
| gpt_oss | 8 | 4 | 3 | 1 |
| kimi_k2 | 16 | 9 | 7 | 0 |

## Representative per-row errors

| model | token | category | error |
|---|---|---|---|
| deepseek_v3 | 1 | runtime | RuntimeError: Unsupported scales/output dtype! |
| deepseek_v3 | 16 | correctness | AssertionError: accuracy check failed: checkAllclose err=0.9969395399093628, logits_diff=0 |
| deepseek_v4 | 1 | runtime | RuntimeError: Unsupported scales/output dtype! |
| deepseek_v4 | 16 | correctness | AssertionError: accuracy check failed: checkAllclose err=0.9969221353530884, logits_diff=1 |
| kimi_k2 | 1 | runtime | RuntimeError: Unsupported scales/output dtype! |
| kimi_k2 | 16 | correctness | AssertionError: accuracy check failed: checkAllclose err=0.9965384602546692, logits_diff=0 |
| gpt_oss | 256 | pass |  |
| gpt_oss | 512 | correctness | AssertionError: accuracy check failed: checkAllclose err=0.9967130422592163, logits_diff=0 |
| gpt_oss | 4096 | runtime | TypeError: __init__(): incompatible function arguments. The following argument types are s |

Source: `docs/baseline_523ca1c7_a8w4_strict.csv` (per-row strict_error, error_category,
aot_status, flydsl_command, kernel-path metrics). aot_status=no_aot for all a8w4: no aiter
AOT cache entry exists for these a8w4 shapes, so the strict runner runs without the AOT
gate; the kernel still compiles+runs and then fails the strict correctness gate or a runtime
scale/output check -- a real correctness/runtime block, not merely a missing AOT artifact.
