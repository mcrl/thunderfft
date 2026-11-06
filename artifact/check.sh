#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
ARTIFACT_PYTHON="${THUNDERFFT_PYTHON:-${PYTHON:-python3}}"
"${ARTIFACT_PYTHON}" -m unittest discover -s artifact/tests -v
"${ARTIFACT_PYTHON}" -m unittest discover -s artifact/cloud -p 'test_*.py' -v
for suite in artifact/validation/test_evaluate.py artifact/table2/test_model.py \
             artifact/table3/test_measure.py artifact/figure2/test_plot.py \
             artifact/table5/check_summary.py artifact/table7/test_contract.py \
             artifact/table7/test_profile.py artifact/table3/test_assemble.py \
             artifact/figure2/test_assemble.py artifact/table5/test_assemble.py \
             artifact/table7/test_assemble.py artifact/table7/test_utilization.py \
             artifact/figure3/test_measure.py artifact/figure4/test_measure.py \
             artifact/figure5/test_measure.py artifact/table6/test_measure.py; do
  "${ARTIFACT_PYTHON}" "${suite}"
done
for script in run.sh artifact/common.sh artifact/setup.sh artifact/*/run.sh; do
  bash -n "${script}"
done
"${ARTIFACT_PYTHON}" -m json.tool artifact/registry.json >/dev/null
echo "Artifact CPU checks and shell syntax checks passed."
