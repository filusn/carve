"""Input-removal oracle = the achievable-recovery ceiling (Phase 6).

Every activation-space method (SAE, raw-neuron, CAV, …) tries to reproduce the gold input
effect e_in = f(x_art) − f(remove(x_art)) by editing block-ℓ activations. The oracle instead
removes the artifact at the INPUT — which by construction recovers e_in exactly. It is not a
deployable method (you can't remove an unknown real artifact at test time); it is the
reference row that tells you what recovery was attainable, so a method's R is read against
1.0, not against nothing.

The harness models this with ``run_cell(..., oracle=True)``: removing the artifact maps
x_art→clean and leaves clean unchanged, so both intervened decisions equal f(clean)=f_removed
⇒ R≡1, Selectivity≡1. This module documents the semantics and carries the canonical meta.
"""
from __future__ import annotations

ORACLE_META = {"selection": "oracle", "method": "input_remove"}
