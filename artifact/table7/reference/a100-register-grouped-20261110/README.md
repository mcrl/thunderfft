# Register permutation branches grouped by stage on A100

Measured on 2026-11-10, A100 80GB PCIe, N=64 TF32, batch=65536 and R=1000.
The final selected `register_branch_cached` candidate measures **6.5413 µs**
through the normal runner. The native matched comparison measures **6.5403 µs**
versus **9.3051 µs** before grouping: approximately **29.7% less time**.
These are repeated-computation measurements of a derived complete FFT, not
full FFT launch latency or an identified original-paper implementation.

## Change and matched comparison

The previous candidate called the checkpoint's three conditional-swap groups
separately for each of eight register quartets. The optimized candidate checks
each condition once per stage, applying its swaps across all eight disjoint
quartets before the next condition. The ordering within each quartet is
unchanged. Lane-dependent branches remain; no stage permutation, operand or
MMA is omitted. The `51e346c` input/output mapping, initialized 28-pair
coefficient cache and 48 MMA operations remain the same.

| Native comparison candidate | Median time (µs) |
|---|---:|
| Previous per-quartet three-condition candidate | 9.305088043 |
| Selected three conditions applied across each stage | 6.540287972 |
| Alternative single switch applied across each stage | 6.658047676 |

The selected three-condition version preserves the checkpoint's conditional
swap formulation and is faster in this comparison than the stage-wide switch
alternative. Both alternatives pass the separate R=1 accuracy check.

**Timing boundary:** both baseline and optimized candidates prepare their 28
coefficient pairs once before repetition. The `(t_2R - t_R) / R` protocol
excludes this preparation and initial/final memory transfers in both cases.
The optimization does not gain time by omitting work or changing that boundary.
Repeated outputs may overflow and are not used as correctness evidence.

## Final runner measurements

Both final collections use five paper-protocol trials and the same CMake
binary. The selected candidate measures **0.00654131174088 ms** in validated
mode and **0.00654131221771 ms** in default timing-only mode. Timing-only
results remain explicitly unchecked even when a separate validation succeeds.

| Native variant | Validated median (µs) | Maximum per-transform relative L2 at R=1 |
|---|---:|---:|
| `register_branch_cached` | 6.541311741 | 0.00151072429021 |
| `register_branch` | 11.939840317 | 0.00151072429021 |
| `register_uniform` | 3.668992281 | 0.00151327457482 |
| `shared_stage_radix4` | 10.976256371 | 0.00151327457482 |

Independent Gaussian and complex-tone checks use a CPU FP64 DFT reference
and a 0.005 maximum relative L2 gate. The selected candidate retains the
baseline's accuracy. Assembly with compatible retained Table 6 evidence
completes Table 7 with no missing cells; the first row remains an explicit
tcFFT-utilization estimate, not a measured radix-16 kernel.

## Compiled workload and validation

Register use decreases from **194 to 160** without local-memory spills.
The repeated interval retains all **48 HMMA** instructions and contains no
shared, global or local memory instructions. Its static control counts drop
from 16 to **6 BSSY / 6 BSYNC**, and from 33 to **9 BRA** instructions. The
static loop body has **584 instructions**, including divergent alternatives;
these totals are not general dynamic instruction counts. No forward branch
skips an HMMA. The final predicated call reaches the one-time output epilogue
and exits, so that output traffic is outside repeated computation.

Compute Sanitizer reports **0 memory errors** and **0 race hazards** on the
final binary. Source and binary SHA-256 values are:

- Source: `39fbcae9b8a1ab8df3f9436d30108773f8d4beded2f8c8e9706b43df57b55bdb`
- Binary: `31ea081c7f52f301daea69690db508d901581e68328163416c0273804b46f8b9`

`grouping_proof.py` independently proves exact permutation equivalence with
arbitrary symbolic register values. It checks strides 1 and 4, all four lane
patterns, all 24 quartet orderings and disjoint coverage of all 32 registers:
192 cases and 6,144 register-value comparisons for each candidate. This is a
mathematical permutation proof; it does not replace GPU accuracy, sanitizer
or compiled-instruction checks.

```bash
python3 grouping_proof.py --output grouping-proof.json
```

## Provenance and evidence

The selected source remains a reconstruction using the `51e346c` data mapping
and conditional swaps from `652808b`, before `403c6ae` (`reduce divergence`).
The unmodified checkpoint omits coefficient initialization and permutations;
its approximately 3.48 µs partial timing is not a valid FFT measurement.
Original paper-stage identity remains unconfirmed. The previous
[9.30 µs archive](../a100-register-cached-20261110/README.md) remains unchanged;
its older source hashes cannot fill the current selected row.

- `validated/` and `timing-only/`: independent final runner collections.
- `comparison/`: baseline, grouped-conditions and grouped-switch native evidence.
- `audit/`: final SASS, resource usage, instruction audit and sanitizer captures.
- `source/`: final native benchmark source and kernel headers.
- `grouping_proof.py` and `grouping-proof.json`: independent symbolic proof.
- `archive.json`: hashes of the retained evidence.

To import and render without GPU work:

```bash
./run.sh --backend local --profile paper --collection register-grouped \
  --part validated \
  --import-result table7=artifact/table7/reference/a100-register-grouped-20261110/validated
./run.sh table7 --backend local --profile paper \
  --collection register-grouped --assemble
```

The first-row estimate requires compatible Table 6 tcFFT evidence in the same
collection. Import timing-only evidence into a separate collection; conflicting
measurements of the same variant must not be combined. Native comparison and
runner medians remain separate rather than substituting for one another.
