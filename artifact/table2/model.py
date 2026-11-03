#!/usr/bin/env python3
"""Calculate Table 2 model rows; no CUDA or measured cuFFTDx row is used."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def calculate(config):
    for key in ("cuda_peak_tflops", "tensor_peak_tflops", "shared_bytes_per_sm_cycle",
                "tensor_ops_per_sm_cycle", "mma_m", "component_bytes"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    ideal_cost = 5.0 / config["cuda_peak_tflops"]
    rows = [{"gpu": config["gpu"], "precision": config["precision"],
             "origin": "calculated", "engine": "CUDA Core", "design": "Ideal FFT",
             "mma_shape": "", "peak_tflops": config["cuda_peak_tflops"],
             "arithmetic_coefficient": 5.0, "shared_bytes_per_mma": "",
             "mma_issue_cycles": "", "required_shared_bytes_per_cycle": "",
             "utilization_ceiling": 1.0, "relative_time": 1.0}]
    for radix in config["radices"]:
        if radix not in (4, 8, 16):
            raise ValueError("This paper model defines radices 4, 8 and 16 only")
        m, n, k = config["mma_m"], 2 * radix, 2 * radix
        ops = 2 * m * n * k
        issue_cycles = ops / config["tensor_ops_per_sm_cycle"]
        traffic = (m * k + m * n) * config["component_bytes"]
        required_bw = traffic / issue_cycles
        utilization = min(1.0, config["shared_bytes_per_sm_cycle"] / required_bw)
        coefficient = 8.0 * radix / math.log2(radix)
        rows.append({"gpu": config["gpu"], "precision": config["precision"],
                     "origin": "calculated", "engine": "Tensor Core",
                     "design": f"Radix-{radix}", "mma_shape": f"m{m}n{n}k{k}",
                     "peak_tflops": config["tensor_peak_tflops"],
                     "arithmetic_coefficient": coefficient,
                     "shared_bytes_per_mma": traffic, "mma_issue_cycles": issue_cycles,
                     "required_shared_bytes_per_cycle": required_bw,
                     "utilization_ceiling": utilization,
                     "relative_time": coefficient / (config["tensor_peak_tflops"] * utilization) / ideal_cost})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = HERE / "assumptions.json"
    config = json.loads(inputs.read_text())
    rows = calculate(config)
    for name in ("data.csv", "table.csv"):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Table 2: calculated model rows", "",
             "| Design | Arithmetic coefficient | Utilization ceiling | Relative time |",
             "| --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['design']} | {row['arithmetic_coefficient']:.2f} | "
                     f"{100 * row['utilization_ceiling']:.0f}% | {row['relative_time']:.2f} |")
    lines += ["", "The measured cuFFTDx row is excluded. Inputs are the supplied paper's",
              "model assumptions; these are calculations, not fresh GPU measurements.",
              "The radix-8 calculation retains 64/3 rather than rounding it to 21.3."]
    (args.output / "table.md").write_text("\n".join(lines) + "\n")
    metadata = {"evaluation": "table2", "scope": "calculated rows only",
                "full_paper_comparison": False, "excluded": ["measured cuFFTDx row"],
                "assumptions": config, "assumptions_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "formulae": {"arithmetic": "8*r/log2(r) times N*log2(N)",
                             "mma_ops": "2*m*n*k; n=k=2*r",
                             "shared_traffic": "(m*k+m*n)*component_bytes; B remains in registers",
                             "utilization": "min(1, available_shared_B/cycle / required_shared_B/cycle)",
                             "relative_time": "(coefficient/(peak_TFLOPS*utilization))/(5/cuda_peak_TFLOPS)"},
                "scientific_validation": "calculated from declared paper assumptions"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Calculated {len(rows)} model rows: {args.output / 'table.csv'}; cuFFTDx measurement excluded")


if __name__ == "__main__":
    main()
