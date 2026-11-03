#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
ARTIFACT_PYTHON="${THUNDERFFT_PYTHON:-${PYTHON:-python3}}"
"${ARTIFACT_PYTHON}" -m unittest discover -s artifact/tests -v
for suite in artifact/validation/test_evaluate.py artifact/table2/test_model.py \
             artifact/table3/test_measure.py artifact/figure2/test_plot.py \
             artifact/table5/check_summary.py artifact/table7/test_contract.py \
             artifact/table7/test_profile.py artifact/table3/test_assemble.py \
             artifact/figure2/test_assemble.py artifact/table5/test_assemble.py \
             artifact/table7/test_assemble.py; do
  "${ARTIFACT_PYTHON}" "${suite}"
done
for script in run.sh artifact/common.sh artifact/*/run.sh; do
  bash -n "${script}"
done
"${ARTIFACT_PYTHON}" -m json.tool artifact/registry.json >/dev/null
echo "Artifact CPU checks and shell syntax checks passed."
