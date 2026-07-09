#!/usr/bin/env bash
# CARVE — full experiment pipeline on the config-driven artifact set.
#
# Runs the whole grid inside the research-base GPU container against whatever
# artifacts.types is set to in configs/default.yaml (the NEW set = ruler, arrow,
# black_corner). Pipeline order:
#   (a) scripts/01_smoke_test.py      -> Phase-0 bias table  (per-artifact e_in + bias_gap)
#   (b) scripts/40_run_interventions  -> Phase-5 SAE-intervention grid (recovery/selectivity)
#   (c) scripts/50_baselines.py       -> Phase-6 baselines grid (identical ground truth)
#   (d) scripts/60_figures.py         -> Phase-7 figures (from the two NEW grids only)
#
# Every run writes a fresh UTC-timestamped dir under experiments/runs/; nothing
# existing is overwritten or deleted. Figures are written UNDER the new baselines
# run dir so the committed figures/ tree is untouched.
#
# Usage:
#   scripts/run_new_artifact_grid.sh            # FULL 3-seed grid (launch only after PI OK)
#   scripts/run_new_artifact_grid.sh --quick    # fast harness validation (1 seed, small N)
set -euo pipefail

QUICK_ARG=""
for a in "$@"; do
  case "$a" in
    --quick) QUICK_ARG="--quick" ;;
    *) echo "unknown arg: $a (only --quick is accepted)" >&2; exit 2 ;;
  esac
done

IMAGE="research-base:2026-07"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ISIC data mounted READ-ONLY at the SAME absolute path it has on the host, so the
# git-ignored data/ symlinks (data/isic2018 -> /home/barmon/ceai/...) resolve identically
# inside the container. RO honors the no-mutate-source rule.
DATA_SRC="/home/barmon/ceai/datasets/ISIC/ISIC_2018"

docker run --rm \
  --gpus all \
  -v "${REPO_DIR}:/workspace" \
  -v "${DATA_SRC}:${DATA_SRC}:ro" \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  -e HF_HOME=/workspace/weights/hf \
  "${IMAGE}" \
  bash -c '
set -euo pipefail
QUICK="'"${QUICK_ARG}"'"
RUNS=experiments/runs

echo "[deps] pip install -q -r requirements.txt (transformers/omegaconf/matplotlib/pandas/pyarrow ...)"
pip install -q -r requirements.txt

# newest run dir whose name ends in the given suffix (each script just created one)
newest() { ls -1dt ${RUNS}/*_"$1" 2>/dev/null | head -1; }

echo "================ (a) Phase-0 bias table (scripts/01_smoke_test.py) ================"
python3 scripts/01_smoke_test.py ${QUICK}
PHASE0=$(newest phase0_gate); echo "[run-dir] phase0        -> ${PHASE0}"

echo "================ (b) Phase-5 interventions grid (scripts/40_run_interventions.py) ================"
python3 scripts/40_run_interventions.py ${QUICK}
INTERV=$(newest interventions_grid); echo "[run-dir] interventions -> ${INTERV}"

echo "================ (c) Phase-6 baselines grid (scripts/50_baselines.py) ================"
python3 scripts/50_baselines.py ${QUICK}
BASE=$(newest baselines_grid); echo "[run-dir] baselines     -> ${BASE}"

echo "================ (d) Phase-7 figures (scripts/60_figures.py) ================"
# Aggregate ONLY the two NEW grids (pass their common-nothing parent is unsafe, so we pool
# the baselines grid — the complete apples-to-apples method set, as in the Stage-6 precedent)
# and write PNGs UNDER the baselines run dir so committed figures/ are never overwritten.
python3 scripts/60_figures.py --runs "${BASE}" --out "${BASE}/figures"
echo "[run-dir] figures       -> ${BASE}/figures"

echo
echo "==================== NEW RUN DIRS ===================="
echo "  phase0 bias table : ${PHASE0}"
echo "  interventions grid: ${INTERV}"
echo "  baselines grid    : ${BASE}"
echo "  figures           : ${BASE}/figures"
echo "======================================================"
'
