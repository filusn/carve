"""ISIC-2018 Task 3 (HAM10000) loader for the binary melanoma-vs-nevus task.

The images are provisioned locally under ``paths.data_root`` (symlinked, git-ignored) — no
download step. This is the **reservoir canvas** from docs/DATASETS.md: HAM10000 is used as
the injection canvas because the Bissoto artifact annotations (the "verified-clean" filter
and the real-artifact slice) are not available in this environment. See docs/LOG.md for the
canvas decision and its consequences (no Bissoto clean-pool filter / no Bissoto real slice).

Model-agnostic: returns image ids, file paths, and binary labels, plus a ``load_image``
that yields float32 HxWx3 in [0,1] — exactly the canvas ``carve.data.artifacts.inject``
expects. Encoder-specific preprocessing lives in ``carve.models.encoders``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

# ISIC-2018 Task 3 one-hot label columns (docs/DATASETS.md)
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


@dataclass
class ISICBinary:
    """A binary derm dataset: label 1 = positive class (e.g. MEL), 0 = negative (e.g. NV)."""

    ids: list[str]
    paths: list[str]
    labels: np.ndarray  # int {0,1}
    pos_class: str
    neg_class: str
    image_size: int | None = None

    def __len__(self) -> int:
        return len(self.ids)

    def load_image(self, i: int, size: int | None = None) -> np.ndarray:
        """Load image `i` as float32 HxWx3 in [0,1] (optionally resized to a square)."""
        img = Image.open(self.paths[i]).convert("RGB")
        s = size if size is not None else self.image_size
        if s is not None:
            img = img.resize((s, s), Image.BILINEAR)
        return (np.asarray(img, dtype=np.float32) / 255.0)

    def summary(self) -> dict:
        n_pos = int(self.labels.sum())
        return {
            "task": f"{self.pos_class}_vs_{self.neg_class}",
            "n": len(self),
            "n_pos": n_pos,
            "n_neg": len(self) - n_pos,
            "pos_rate": float(self.labels.mean()) if len(self) else 0.0,
            "pos_class": self.pos_class,
            "neg_class": self.neg_class,
        }


def _resolve(cfg, root, image_dir, groundtruth_csv):
    """Resolve dir/csv from explicit args → cfg.dataset.* → ISIC-2018 Task3 defaults.

    Config-driven (CLAUDE.md): reads `dataset.root`, `dataset.image_subdir`, and
    `dataset.groundtruth_csv` (the latter two relative to `root` unless absolute).
    """
    ds = cfg.dataset if cfg is not None else None
    if root is None:
        root = (ds.get("root") if ds is not None else None)
        if root is None:
            data_root = (cfg.paths.data_root if cfg is not None else "data") or "data"
            root = os.path.join(str(data_root), "isic2018")
    root = str(root)
    if image_dir is None:
        subdir = (ds.get("image_subdir") if ds is not None else None) \
            or "ISIC2018_Task3_Training_Input"
        image_dir = subdir if os.path.isabs(subdir) else os.path.join(root, subdir)
    if groundtruth_csv is None:
        gt = (ds.get("groundtruth_csv") if ds is not None else None) or os.path.join(
            "ISIC2018_Task3_Training_GroundTruth", "ISIC2018_Task3_Training_GroundTruth.csv"
        )
        groundtruth_csv = gt if os.path.isabs(gt) else os.path.join(root, gt)
    return root, image_dir, groundtruth_csv


def load_isic_binary(
    cfg=None,
    pos_class: str | None = None,
    neg_class: str | None = None,
    root: str | None = None,
    image_dir: str | None = None,
    groundtruth_csv: str | None = None,
    image_size: int | None = None,
) -> ISICBinary:
    """Load the ISIC-2018 Task3 binary subset (default: melanoma MEL=1 vs nevus NV=0).

    Config-driven: `pos_class`/`neg_class`/`image_size` and the paths fall back to
    `cfg.dataset.*` (then to sensible defaults) when not passed explicitly. Keeps only rows
    whose one-hot label is exactly `pos_class` or `neg_class`, and only images that exist on
    disk. Deterministic ordering (sorted by isic id) so splits built on top are reproducible.
    """
    ds = cfg.dataset if cfg is not None else None
    pos_class = pos_class or (ds.get("pos_class") if ds is not None else None) or "MEL"
    neg_class = neg_class or (ds.get("neg_class") if ds is not None else None) or "NV"
    if image_size is None:
        image_size = int(ds.get("image_size", 224)) if ds is not None else 224
    _, image_dir, groundtruth_csv = _resolve(cfg, root, image_dir, groundtruth_csv)
    if not os.path.exists(groundtruth_csv):
        raise FileNotFoundError(f"ISIC ground-truth CSV not found: {groundtruth_csv}")

    df = pd.read_csv(groundtruth_csv)
    id_col = "image" if "image" in df.columns else df.columns[0]
    keep = (df[pos_class] == 1) | (df[neg_class] == 1)
    df = df[keep].copy()
    df["_label"] = (df[pos_class] == 1).astype(int)
    df = df.sort_values(id_col).reset_index(drop=True)

    ids, paths, labels = [], [], []
    for _, row in df.iterrows():
        iid = str(row[id_col])
        p = os.path.join(image_dir, iid + ".jpg")
        if os.path.exists(p):
            ids.append(iid)
            paths.append(p)
            labels.append(int(row["_label"]))

    if not ids:
        raise FileNotFoundError(
            f"no {pos_class}/{neg_class} images found under {image_dir}; check the mount"
        )
    return ISICBinary(
        ids=ids,
        paths=paths,
        labels=np.array(labels, dtype=int),
        pos_class=pos_class,
        neg_class=neg_class,
        image_size=image_size,
    )
