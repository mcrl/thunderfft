# Derived register FFT with checkpoint permutations on A100

Measured on 2026-11-10, A100 80GB PCIe. The native comparison uses N=64,
TF32, batch=65536 and R=1000. The selected `register_branch_cached` candidate
measures **9.3041 µs** through the final runner, in both validated and default
timing-only modes; the separate native comparison measures **9.3051 µs**.
It is a newly derived complete FFT, not an unmodified historical checkpoint
or a recovered original-paper experiment.

## Implementation and historical source

The implementation retains the valid `51e346c` register data mapping and
MMA helper. Its stage permutation uses the three conditional-swap groups
preserved in checkpoint `652808b`, before `403c6ae` (`reduce divergence`).
These swaps implement the same permutation as the older four-way switch.
Every stage permutation and MMA operand is present. All 28 distinct coefficient
pairs are initialized and cached once before the repetition loop.

The unmodified `652808b` FP32 checkpoint comments out coefficient initialization
and all inter-stage permutations. Its retained approximately 3.48 µs timing
therefore describes a partial instruction workload, not a valid FFT. Restoring
these operations changes that workload. Neither that timing nor the partial
`e4aa1b0` timing supplies the selected paper row.

The derived kernel keeps the `51e346c` shared-memory input/output layout, not
the checkpoint's different padding. Those layouts have the same logical
register input mapping. Within the repeated interval, data stays in registers
and the three-condition permutations exchange register values between stages.
The source remains `implementation_kind=reconstructed`, with no invented
historical source commit. Source and binary hashes identify the exact candidate.

## Final runner measurements

Both collections use five paper-protocol trials (batch=65536, R=1000) and the
same final CMake binary. Validated mode uses three warmups per repetition
count and bounded Gaussian timing input; timing-only mode uses one warmup
and constant input, with accuracy explicitly unchecked.

| Native variant | Validated median (µs) | Timing-only median (µs) |
|---|---:|---:|
| `register_branch_cached` | 9.304064751 | 9.304063797 |
| `shared_stage_radix4` | 10.974207878 | 10.974208832 |
| `register_branch` | 11.939840317 | 11.938816071 |
| `register_uniform` | 3.668992043 | 3.668992281 |

The selected derived register candidate is approximately 15.2% faster than
the shared-stage candidate under their common prepared-once coefficient
boundary. The first paper row remains an explicit estimate from compatible
Table 6 tcFFT evidence, not a measured radix-16 kernel.

## Native comparison

| Candidate | Median time (µs) |
|---|---:|
| Original register branch (`51e346c`) | 11.9388 |
| Restored checkpoint with coefficient setup in repetition | 11.4207 |
| Earlier derived switch-based branch with 28 cached pairs | 10.3741 |
| Selected derived checkpoint permutation with 28 cached pairs | 9.3051 |
| Shared-stage candidate with 21 cached pairs | 10.9742 |
| Uniform register candidate (`5e37821`) | 3.6680 |

These are native comparison measurements; normal runner collections retain
their own medians and trials. The two cached register candidates differ in
permutation implementation. Relative to the earlier cached switch candidate,
the three-condition candidate reduces measured time by approximately 10.3%.
Historical and experimental diagnostic measurements remain separate from the
selected source; they are not interchangeable collection parts.

## Historical restoration checks

| Historical source / diagnostic change | R=1 correctness | Median time (µs) |
|---|---|---:|
| `c800961` (`dump`) / `e4aa1b0` (`float implementation done`): restore only B initialization | Failed; max relative L2 1.15177856249 | — |
| Same path: restore B initialization and correct checkpoint three-condition permutation | Passed | 12.8645 |
| `652808b` (`ckpt`): restore coefficient preparation and stage permutations | Passed | 11.4207 |
| `bbeffb2` (`dump`): restore B initialization, scalar offsets and eight-scalar permutation | Passed | 11.4452 |
| Selected derived implementation: checkpoint permutation and 28 cached coefficient pairs | Passed | 9.3051 |

Inspection found the same relevant executable path in `c800961` and `e4aa1b0`,
so one representative implementation was measured. The original files and
older archives remain unchanged. Every repaired row above describes a derived
restoration, not the unmodified historical commit.

The B-initialization-only diagnostic failed independent R=1 Gaussian and tone
checks, so the accuracy gate stopped execution **before timing**. Its blank
time is deliberate; the maximum relative L2 of 1.15178 must not be interpreted
as a valid FFT timing. Correcting the permutation as well passes the check.
These diagnostic results explain why commit names or a favorable timing alone
cannot establish the paper row's correctness or source identity.

**Timing boundary:** the reported `(t_2R - t_R) / R` excludes coefficient
preparation, initial input transfer and final output transfer. Both the selected
register candidate and `shared_stage_radix4` explicitly prepare coefficients
before repetition. These timings measure repeated unit-kernel computation,
not full FFT launch latency. The original register branch and the restored
uncached checkpoint retain coefficient setup in their repeated source path;
comparisons to them include that change in timing scope.

## Correctness and compiled workload

Separate R=1 Gaussian and complex-tone checks pass against a CPU FP64 DFT.
Maximum per-transform relative L2 error is **0.00151072429021** (gate 0.005).
Repeated timing outputs may overflow and are not used as correctness evidence.
Default timing-only collections remain explicitly unchecked even when a
separate validation run passes.

The selected compiled candidate uses **194 registers**, no local-memory spills,
and **48 HMMA instructions per repeat**. The repeated interval contains no
`LDG`, `LDS`, `STS`, `LDL` or `STL` instructions. Its Tensor Core arithmetic count
is 6,144 FLOPs per FFT. Initial and final memory operations remain outside that
interval. The host W64 LUT follows the shared/uniform candidate construction;
the archived original branch's float-PI construction is unchanged.

Compute Sanitizer on the final CMake binary reports **0 memory errors** and
**0 race hazards**. The binary SHA-256 is
`b92447d14da336e671303de0ac1b65829c562f280a6b1e4de1b1ba94c1d30278`,
matching both runner collections and the SASS audit.

The coefficient proof checks 1,536 pair positions (3,072 scalar components)
across all lanes, stages and butterflies, establishing exact LUT index, sign
and component-order equality for all 28 cached pairs without assuming
trigonometric symmetry. The permutation proof checks 96 symbolic cases.
The SASS audit traces the final predicated call to a one-time output epilogue
that reaches `EXIT`; it does not return to the repeated computation. That
epilogue contains the expected final shared/global output operations.

## Evidence organization

`comparison/` retains native comparisons and candidate source snapshots.
`validated/` and `timing-only/` are independent runner collections: only
matching source/dependency hashes can populate the selected paper row.
`comparison/float-done-init`, `float-done-restored`, `ckpt-restored`,
`ckpt-cached` and `bbeffb2-restored` retain the historical diagnostics, their
source snapshots and native outputs. Failed validation remains visible.
`coefficient-mapping-proof.json` and `verify_coefficient_mapping.py` retain the
CPU proof. `ckpt-cached*.sass` and `ckpt-cached-sass-audit.json` retain the
compiled-workload audit. `final-memcheck*` and `final-racecheck*` preserve
sanitizer and native validation captures. `archive.json` records all retained
evidence hashes. Runner medians and native diagnostic medians remain distinct.

To import a runner collection and render without GPU work:

```bash
./run.sh --backend local --profile paper --collection register-cached \
  --part validated \
  --import-result table7=artifact/table7/reference/a100-register-cached-20261110/validated
./run.sh table7 --backend local --profile paper \
  --collection register-cached --assemble
```

The first-row estimate also requires compatible Table 6 tcFFT evidence in the
collection. Import `timing-only/` into a separate collection; do not combine
conflicting measurements of the same native variant. Older source hashes,
including the earlier switch-based cached candidate, cannot fill the current
selected row. Original paper-stage identities remain unconfirmed.
