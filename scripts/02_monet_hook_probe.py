#!/usr/bin/env python
"""Stage-1 MONET hook probe — the gating check for CARVE's whole SAE/intervention arm.

The intervention design (docs/EXECUTION_PLAN.md Phase 5, docs/BUILD_ORDER.md Stage 1)
assumes we can READ and WRITE MONET's ViT residual-stream activations at a chosen layer.
This script proves that end-to-end on the real model, with ZERO project code beyond
carve.utils, and records what the rest of the pipeline needs to know:

    * how MONET loads (HF `chanwkim/monet`, a CLIP ViT-L/14) and its module tree
    * the residual-stream tensor shape at layer ℓ  (→ SAE input dim / width / k)
    * that a forward hook can READ activations at ℓ
    * that a hook can WRITE (ablate / steer) at ℓ and MOVE the zero-shot decision
      → the mechanism every carve.interventions op will use
    * that the config `model.layer_sweep` indices are valid for MONET's depth

Run (inside the Docker dev container):
    docker exec carve-dev python3 scripts/02_monet_hook_probe.py

MONET weights (~1.7 GB) download from HF on first run into $HF_HOME (=/workspace/weights/hf
in carve-dev). If offline, the script says so and exits without inventing a result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from carve.utils import device, load_config, new_run_dir, save_json  # noqa: E402

MONET_HF_ID = "chanwkim/monet"  # CLIP ViT-L/14, ~105k derm pairs (Kim et al. Nat.Med.2024)


def _find_vision_layers(model):
    """Return (layers_ModuleList, dotted_path) for the ViT encoder blocks, robust to the
    transformers version's attribute layout (CLIPModel.vision_model.encoder.layers)."""
    for path in ("vision_model.encoder.layers", "encoder.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj, path
        except AttributeError:
            continue
    # last resort: search
    for name, mod in model.named_modules():
        if name.endswith("encoder.layers"):
            return mod, name
    raise RuntimeError("could not locate the ViT encoder layers in the MONET module tree")


def main() -> None:
    cfg = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    dev = device(cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 0))
    run = new_run_dir(cfg.paths.runs_dir, "monet_hook_probe", cfg, seed)
    report: dict = {"monet_hf_id": MONET_HF_ID, "device": dev}

    print("=" * 72)
    print("CARVE — Stage-1 MONET hook probe")
    print("=" * 72)
    print(f"[env] device = {dev}")

    # --- 1. load MONET (frozen) ------------------------------------------------------
    try:
        from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

        print(f"[load] {MONET_HF_ID} (downloads ~1.7 GB on first run into $HF_HOME) ...")
        processor = AutoProcessor.from_pretrained(MONET_HF_ID)
        model = AutoModelForZeroShotImageClassification.from_pretrained(MONET_HF_ID)
        model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)  # encoder is FROZEN (CLAUDE.md)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        print(f"[FAIL] could not load MONET: {msg}")
        print("       If this is a network/offline error, download weights then re-run.")
        save_json(run / "probe_report.json", {**report, "status": "load_failed", "error": msg})
        print(f"[run] {run}")
        sys.exit(2)

    layers, layers_path = _find_vision_layers(model)
    n_layers = len(layers)
    hidden = int(getattr(model.config.vision_config, "hidden_size", 0)) or None
    report.update(n_vision_layers=n_layers, hidden_size=hidden, layers_path=layers_path)
    print(f"[model] vision blocks: {n_layers}  |  hidden dim: {hidden}  |  path: {layers_path}")

    layer_sweep = list(cfg.model.get("layer_sweep", []))
    bad = [l for l in layer_sweep if l < 0 or l >= n_layers]
    report["layer_sweep"] = layer_sweep
    report["layer_sweep_valid"] = not bad
    print(f"[cfg ] layer_sweep {layer_sweep} valid for depth {n_layers}: {not bad}"
          + (f"  (out of range: {bad})" if bad else ""))

    ell = min(int(cfg.model.get("probe_layer", n_layers // 2)), n_layers - 1)
    report["probe_layer"] = ell

    # --- 2. build a dummy 2-image batch + zero-shot mel/nevus prompts ----------------
    rng = np.random.default_rng(seed)
    dummy = [(rng.uniform(0.3, 0.8, (224, 224, 3)) * 255).astype("uint8") for _ in range(2)]
    prompts = [cfg.model.zero_shot_prompts.pos, cfg.model.zero_shot_prompts.neg]
    inputs = processor(images=dummy, text=prompts, return_tensors="pt", padding=True).to(dev)

    captured: dict = {}

    def read_hook(_m, _inp, out):
        # CLIP encoder layer returns a tuple; hidden state is out[0]: [B, tokens, hidden]
        captured["act"] = (out[0] if isinstance(out, tuple) else out).detach()

    # --- 3. READ activations at ℓ ----------------------------------------------------
    h = layers[ell].register_forward_hook(read_hook)
    with torch.no_grad():
        base = model(**inputs)
    h.remove()
    act = captured["act"]
    report["residual_shape_at_layer"] = list(act.shape)
    print(f"[READ] activation at layer {ell}: shape {tuple(act.shape)} (B, tokens, hidden)")

    # zero-shot decision signal = logit margin z_pos - z_neg (per image)
    def margin(outputs):
        lpi = outputs.logits_per_image  # [n_images, n_texts]
        return (lpi[:, 0] - lpi[:, 1]).detach().cpu()

    base_margin = margin(base)
    report["base_logit_margin"] = base_margin.tolist()
    print(f"[base] zero-shot logit margin (pos-neg): {base_margin.tolist()}")

    # --- 4. WRITE: ablate (zero) the residual stream at ℓ ----------------------------
    def ablate_hook(_m, _inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + tuple(out[1:])
        return torch.zeros_like(out)

    h = layers[ell].register_forward_hook(ablate_hook)
    with torch.no_grad():
        abl = model(**inputs)
    h.remove()
    abl_margin = margin(abl)
    d_ablate = (abl_margin - base_margin).abs().max().item()
    report["ablate_max_abs_margin_shift"] = d_ablate
    print(f"[WRITE/ablate] margin -> {abl_margin.tolist()}  (max |Δ| = {d_ablate:.4f})")

    # --- 5. WRITE: steer (add a vector) at ℓ -----------------------------------------
    steer_vec = torch.randn(act.shape[-1], device=dev) * float(act.std())

    def steer_hook(_m, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + steer_vec,) + tuple(out[1:])
        return out + steer_vec

    h = layers[ell].register_forward_hook(steer_hook)
    with torch.no_grad():
        st = model(**inputs)
    h.remove()
    d_steer = (margin(st) - base_margin).abs().max().item()
    report["steer_max_abs_margin_shift"] = d_steer
    print(f"[WRITE/steer]  max |Δmargin| = {d_steer:.4f}")

    # --- verdict ---------------------------------------------------------------------
    writable = (d_ablate > 1e-4) or (d_steer > 1e-4)
    report["read_ok"] = True
    report["write_ok"] = bool(writable)
    report["status"] = "GO" if writable and report["layer_sweep_valid"] else "REVIEW"
    save_json(run / "probe_report.json", report)

    print("-" * 72)
    print(f"[verdict] READ ok: True | WRITE moves decision: {writable} "
          f"| layer_sweep valid: {report['layer_sweep_valid']}")
    if report["status"] == "GO":
        print("[GO] MONET activations are hook-readable AND hook-writable via HF forward "
              "hooks.\n     carve.interventions can be built on this path (no Prisma "
              "dependency required).")
    else:
        print("[REVIEW] read worked but writing didn't move the decision, or config layers "
              "are out of range — inspect before building the intervention layer.")
    print(f"[run] {run}")


if __name__ == "__main__":
    main()
