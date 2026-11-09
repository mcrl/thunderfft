# Optimized shared-memory radix-4 reconstruction on A100

Measured on 2026-11-10 (KST), A100 80GB PCIe, CUDA compiler 12.9.41 /
runtime 12.8. N=64, TF32, batch=65536, R=1000. No GPU clocks or power limits
were changed. This is a new reconstruction; original paper-source identity
remains unconfirmed.

**The final default measures 0.0109742 ms**, below the 0.0120 ms target.
The separate validated-mode result is 0.0109763 ms. Both use three trials
through `run.sh table7 --backend local --profile paper`. A separate five-trial
check of the final rebuilt binary measures 0.0109773 ms.

| Matched candidate | Median time (µs) | Trials |
|---|---:|---:|
| Previous default (`d404279` source) | 22.8014 | 5 |
| Bank layout, vector access and scheduling; coefficients inside repeat | 15.5013 | 3 |
| Coefficients cached before repeat, untiled butterflies | 12.7519 | 5 |
| Cached coefficients and four-butterfly scheduling | 10.9752 | 5 |

The final change is about **2.08× faster** than the previous default. Its
five-trial candidate range was 10.975231–10.978303 µs. The untiled cache
candidate was less stable (12.391424–13.069312 µs); all raw trials are retained.
`comparison/` contains unchanged native CSV/logs, source snapshots, build
defines and source/binary hashes. These are diagnostic comparisons, not
interchangeable collection parts. Historical headers require the unchanged
helper headers and benchmark from the recorded repository source.

The implementation keeps every stage's shared-memory exchange and uses no
inter-stage register permutation. Its fixed XOR address layout, aligned
`float2` access and four independent butterflies per scheduling group reduce
bank conflicts and dependency stalls. Shared storage drops from 9,344 to
8,192 bytes. Compiled code retains **48 HMMA, 96 LDS.64 and 96 STS.64** per warp
per repeat, with no local-memory spills. The arithmetic count remains
6,144 Tensor Core FLOPs per FFT.

**Timing boundary:** 21 invariant MMA coefficient pairs are prepared once per
kernel launch, before the repeat loop. The existing `(t_2R - t_R) / R` protocol
excludes that preparation, initial loading and final output. This measures
repeated unit-kernel computation, not a full FFT call's latency. The gain
includes moving invariant preparation outside the repeated interval; it is
not solely a shared-layout improvement. CSV/metadata record
`timing_scope=repeated_kernel_compute_coefficients_prepared_once`, and the
rendered Table 7 `‡` note states the boundary.

Validation of the final binary:

- Separate R=1 bounded complex Gaussian and complex-tone inputs pass against
  a CPU FP64 DFT. Maximum per-transform relative L2 error is
  **0.00151327457482**, unchanged from the previous implementation (gate 0.005).
- Finite-output and guard checks pass. Repeated timing outputs may overflow
  and are not used as correctness evidence.
- Compute Sanitizer reports **0 memory errors and 0 race hazards**.
- An independent exhaustive address proof checks the 2,048-word bijection,
  vector alignment and bank uniqueness; it does not simulate GPU execution.
- Final assembly with retained Table 6 evidence has no missing paper cells
  and generates HTML, PDF and PNG. The first row remains an explicit estimate.

`validated/` preserves the three complete candidates' collection files;
`timing-only/` preserves all five default candidates. Default timings remain
unchecked even though the separate validated run passes. `archive.json`
checksums the evidence without modifying the original measurement files.

Import and render the default result without GPU work:

```bash
./run.sh --backend local --profile paper --collection shared-optimized \
  --part timing --import-result table7=artifact/table7/reference/a100-shared-optimized-20261110/timing-only
./run.sh table7 --backend local --profile paper --collection shared-optimized --assemble
```

The baseline estimate requires compatible Table 6 tcFFT evidence in the same
collection. Import `validated/` into a separate collection to inspect its
validated results; do not combine different measurements of the same variant.
The older 22.8 µs reference bundle remains archived, but its source hash no
longer matches the current selected kernel and cannot fill the current row.
