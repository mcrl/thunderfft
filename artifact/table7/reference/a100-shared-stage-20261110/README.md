# A100 validation of the shared-memory radix-4 reconstruction

Measured on 2026-11-10 (KST) with an A100 80GB PCIe, CUDA compiler 12.9.41
and runtime 12.8. This is a new implementation, not recovered original paper
source. Each butterfly reads/writes shared memory, and the three radix-4
stages use no register-fragment permutation. Initial input loading still uses
base-4 digit reversal. One warp processes 16 transforms with 9,344 B of static
shared memory. The compiled kernel contains 48 HMMA instructions per repeat,
accounting for 6,144 Tensor Core FLOPs per FFT.

| Setting / result | Value |
|---|---|
| Workload | N=64, TF32, batch=65536, R=1000, three trials |
| Timing | median of `(t_2R - t_R) / R` |
| Validated-mode median | **22.8055019379 µs** |
| Default timing-only median | **22.7942409515 µs** |
| Paper reference | 12.0 µs; this reconstruction is about 90% slower |
| Maximum per-transform relative L2 error | 0.00151327457482 (gate: 0.005) |
| Memory / race checks | Compute Sanitizer: 0 errors / 0 hazards |

Correctness uses separate R=1 bounded complex Gaussian and complex-tone inputs
against a CPU FP64 DFT, with finite-output and guard checks. Repeated timing
outputs are not used for correctness. The default timing-only run skips those
checks and remains explicitly unchecked, even though the separate validated
run passed. The initial mathematical index proof covers 128 real/imaginary
basis vectors and one random vector; it does not model TF32 rounding or GPU
memory ordering.

A short matched comparison measured the older reconstruction at 25.6256 µs,
the new two-buffer candidate at 22.8127 µs, and the single-buffer candidate at
22.7994 µs. The two new timings are effectively equal; the single-buffer
implementation was selected for its smaller shared-memory allocation. These
exploratory timings are separate from the final result bundles.

`validated/` preserves the three complete candidates' measurement files;
`timing-only/` preserves all five default candidates. Both were collected
through `run.sh table7 --backend local --profile paper`. Files retain native
IDs, source/binary hashes, device identity, command lines and accuracy status.
`archive.json` checksums this evidence; generated views are not archived.

Import the default result and render it without GPU work:

```bash
./run.sh --backend local --profile paper --collection shared-stage \
  --part timing --import-result table7=artifact/table7/reference/a100-shared-stage-20261110/timing-only
./run.sh table7 --backend local --profile paper --collection shared-stage --assemble
```

Import `validated/` into a separate collection to inspect validated results.
Do not combine both bundles for the same variant: their timing protocols and
accuracy status differ. The first-row estimate requires compatible Table 6
tcFFT evidence in the collection; it is never filled with the paper number.
The new second row is marked `‡` as reconstructed; timing-only rows also retain
`†`. In the final local assembly with retained Table 6 evidence, all four paper
rows rendered to HTML, PDF and PNG with no missing cells.
