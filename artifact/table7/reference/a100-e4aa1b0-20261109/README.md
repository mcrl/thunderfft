# A100 confirmation of the selected e4aa1b0 source

Measured on 2026-11-09, using the unchanged historical kernel from
`e4aa1b0bf7d38493fcb8d5ff7c1ffd599c7d6375`.

| Setting / result | Value |
|---|---|
| GPU | A100 80GB PCIe |
| CUDA compiler / runtime | 12.9.41 / 12.8 |
| Workload | N=64, TF32, batch=65536, R=1000 |
| Timing protocol | Constant input, one warmup per repeat count, three trials |
| Median time | 0.00788787221909 ms |
| Minimum / maximum | 0.00788787221909 / 0.00789811229706 ms |
| Tensor Core utilization | 32.7225147943%, against 156 TFLOPS |
| Rounded table cells | **0.0079 ms / 33%** |

This is the selected source for the `+ reducing communication` artifact row.
The kernel omits MMA B-fragment initialization and retains a shared-memory
round trip inside each iteration. Numerical correctness was not checked;
these are partial-kernel instruction timings, not complete-FFT measurements.
Selection for the artifact does not independently establish the original paper
experiment's identity.

The native CSV contains all three trials. `metadata.json` records source and
binary hashes, GPU identity, compiler settings and the included operations.
`archive.json` checksums the copied result bundle without changing its bytes.

To import and render this measurement without running the GPU:

```bash
./run.sh --backend local --profile paper --collection selected-e4aa \
  --part a100-e4aa1b0 \
  --import-result table7=artifact/table7/reference/a100-e4aa1b0-20261109
./run.sh table7 --backend local --profile paper --collection selected-e4aa --assemble
```

Uncollected rows remain blank. A tcFFT baseline estimate requires compatible
Table 6 measurements in the same collection.
