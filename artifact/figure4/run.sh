#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
artifact_init figure4
artifact_pending "Figure 4 needs a correctness-gated convolution adapter: validate the corrected 4096 path on the paper workload, align circular/linear convolution and filter orientation across all baselines, and compare shared inputs before timing. See artifact/figure4/README.md."
