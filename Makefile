# CARVE convenience targets. Real runs require a CUDA GPU. See CLAUDE.md.
.PHONY: setup smoke test lint format inject probe sae interventions baselines aggregate clean

setup:           ## create venv + install deps
	uv venv && . .venv/bin/activate && uv pip install -r requirements.txt && uv pip install -e .

smoke:           ## Phase-0 day-1 go/no-go
	python scripts/01_smoke_test.py

inject:          ## Phase 1 — materialize injected datasets
	python scripts/10_inject.py --config configs/default.yaml

probe:           ## Phase 2 — train probe, measure bias gap + input effect
	python scripts/20_train_probe.py --config configs/default.yaml

sae:             ## Phase 3 — train/load SAE, report health
	python scripts/30_train_sae.py --config configs/default.yaml

interventions:   ## Phases 4-5 — discover features, run interventions + metrics
	python scripts/40_run_interventions.py --config configs/default.yaml

baselines:       ## Phase 6 — CAV/CDEP/raw-neuron/random/oracle
	python scripts/50_baselines.py --config configs/default.yaml

aggregate:       ## Phase 7 — aggregate runs, make tables + figures
	python scripts/60_aggregate.py --config configs/default.yaml

test:            ## run unit tests
	pytest

lint:            ## ruff check
	ruff check .

format:          ## ruff format
	ruff format .

clean:           ## remove caches (keeps data/ and runs/)
	rm -rf .pytest_cache .ruff_cache **/__pycache__
