# A100 unit-kernel measurement archive

This bundle contains the four selected historical kernel measurements from
2026-11-07: N=64, TF32, batch 65,536, R=1,000, one timing trial and one warmup
per repeat count. Inputs are constant `(0.125, -0.25)`. Accuracy and output
guards were not checked. Source commits, operation coverage, source/binary
hashes and GPU/compiler details are retained in the original metadata and logs.
`archive.json` records hashes of the unchanged result files.

| Source | Time (us) | Issued-MMA utilization | Operation coverage |
| --- | ---: | ---: | --- |
| `51e346c` | 11.93677 | 21.62% | Complete-FFT candidate |
| `e4aa1b0` | 7.88890 | 32.72% | Operand setup omitted; shared-memory round trip |
| `652808b` | 3.48160 | 74.14% | Operand setup and stage permutation omitted |
| `5e37821` | 3.66797 | 70.37% | Complete-FFT candidate |

These are unit-kernel timings with differing operation coverage. The source
IDs do not establish the original paper's stage identities. Utilization uses
6,144 issued MMA FLOPs per input slot and the A100 TF32 peak of 156 TFLOPS.

The optional projection workflow in the [Table 7 instructions](../../README.md)
combines these rows with separately labeled estimates derived from the rounded
Table 6 reference timings. It performs no GPU work.
