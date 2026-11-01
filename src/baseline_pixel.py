"""Pixel-based baseline: engineered features, optional SMOTE, boosted trees.

Why this exists. The U-Net arms vary *how* imbalance is handled inside a
segmentation model. This baseline asks a different question: how much of the
task is solvable without spatial context at all? It is the comparator that makes
the convolutional result interpretable rather than merely good.

Why the features are not just RGB. A baseline handed only three colour channels
would be a straw man, and beating a straw man proves nothing. Buildings here are
separated from bare ground largely by *texture* - roof edges, uniform roof
interiors against speckled soil - so the features include local standard
deviation and gradient magnitude at two scales. The baseline still cannot see
shape or neighbourhood layout, and that is the honest limit being measured.

SMOTE is applied to the training pixels only, never to validation or test. That
sounds obvious but is the commonest way an imbalance experiment is invalidated:
resample before the split and synthetic neighbours of test pixels end up in
training, which can lift reported F1 substantially for no real gain.

    python -m src.baseline_pixel --arm pixel_plain
    python -m src.baseline_pixel --arm pixel_smote
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window
from scipy import ndimage

from .data import load_tiles
from .metrics import ConfusionSweep
from .provenance import fingerprint
from .raster import scale_to_unit, valid_mask_for, validity_rule

FEATURE_NAMES = (
    ["r", "g", "b"]
    + [f"std{w}_{c}" for w in (3, 9) for c in "rgb"]
    + [f"mean{w}_{c}" for w in (9,) for c in "rgb"]
    + ["grad"]
)


def features(patch: np.ndarray) -> np.ndarray:
    """(H, W, 3) in [0, 1] -> (H, W, F) float32 feature stack."""
    chans = [patch[..., c] for c in range(patch.shape[-1])]
    feats = list(chans)

    for w in (3, 9):
        for c in chans:
            mean = ndimage.uniform_filter(c, size=w)
            sq = ndimage.uniform_filter(c * c, size=w)
            feats.append(np.sqrt(np.clip(sq - mean * mean, 0, None)))

    for c in chans:
        feats.append(ndimage.uniform_filter(c, size=9))

    grey = patch.mean(axis=-1)
    gy, gx = np.gradient(grey)
    feats.append(np.hypot(gx, gy))

    return np.stack(feats, axis=-1).astype("float32")


def sample_split(cfg: dict, split: str, per_tile: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw pixels from every tile of a split, with their features and labels."""
    tiles = load_tiles(cfg["processed_dir"], cfg["city"])
    tiles = tiles[tiles["split"] == split]
    if tiles.empty:
        raise SystemExit(f"no tiles in split {split!r}")

    image_dir = Path(cfg["raw_dir"]) / cfg["city"] / "images"
    mask_dir = Path(cfg["processed_dir"]) / cfg["city"] / "masks"
    handles: dict[str, tuple] = {}
    rules: dict[str, str] = {}
    X, y = [], []

    for _, r in tiles.iterrows():
        scene = r["scene"]
        if scene not in handles:
            handles[scene] = (rasterio.open(image_dir / f"{scene}.tif"),
                              rasterio.open(mask_dir / f"{scene}.tif"))
            rules[scene] = validity_rule(handles[scene][0])
        src, msk = handles[scene]
        rule = rules[scene]
        win = Window(int(r["col_off"]), int(r["row_off"]),
                     int(r["patch"]), int(r["patch"]))
        arr = src.read(indexes=[1, 2, 3], window=win)
        patch = scale_to_unit(np.transpose(arr, (1, 2, 0)), src.dtypes[0])
        f = features(patch).reshape(-1, len(FEATURE_NAMES))
        lab = (msk.read(1, window=win) > 0).astype("uint8").ravel()
        valid = valid_mask_for(src, win, rule).ravel()

        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        take = min(per_tile, idx.size)
        pick = rng.choice(idx, size=take, replace=False)
        X.append(f[pick])
        y.append(lab[pick])

    for a, b in handles.values():
        a.close()
        b.close()
    return np.concatenate(X), np.concatenate(y)


def make_classifier(kind: str, seed: int):
    from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier

    if kind == "hist_gb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=None,
            early_stopping=True, validation_fraction=0.1, random_state=seed)
    if kind == "adaboost":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=200, learning_rate=0.5, random_state=seed)
    raise ValueError(f"unknown classifier {kind!r}")


def predict_test_scene(clf, cfg: dict, scene: str, out_tif: Path,
                       sweep: ConfusionSweep) -> None:
    """Score the whole held-out scene tile by tile, stitching probabilities."""
    image = Path(cfg["raw_dir"]) / cfg["city"] / "images" / f"{scene}.tif"
    mask = Path(cfg["processed_dir"]) / cfg["city"] / "masks" / f"{scene}.tif"
    patch = int(cfg["patch"])

    with rasterio.open(image) as src, rasterio.open(mask) as msk:
        rule = validity_rule(src)
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", compress="deflate", nodata=255,
                       tiled=True, blockxsize=256, blockysize=256)
        profile.pop("photometric", None)
        out_tif.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(np.full((src.height, src.width), 255, dtype="uint8"), 1)
            for row in range(0, src.height - patch + 1, patch):
                for col in range(0, src.width - patch + 1, patch):
                    win = Window(col, row, patch, patch)
                    arr = src.read(indexes=[1, 2, 3], window=win)
                    p = scale_to_unit(np.transpose(arr, (1, 2, 0)), src.dtypes[0])
                    f = features(p).reshape(-1, len(FEATURE_NAMES))
                    prob = clf.predict_proba(f)[:, 1].reshape(patch, patch)
                    v = valid_mask_for(src, win, rule)
                    y = msk.read(1, window=win)
                    sweep.update(y, prob, valid=v)
                    q = np.where(v, np.round(prob * 254).astype("uint8"), 255)
                    dst.write(q, 1, window=win)


def run_arm(arm: str, arm_cfg: dict, cfg: dict, stats: dict) -> dict:
    rng = np.random.default_rng(int(cfg["seed"]))
    out_dir = Path(cfg["results_dir"]) / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    per_tile = int(arm_cfg.get("pixels_per_tile", 200))
    X, y = sample_split(cfg, "train", per_tile, rng)
    before = {"n": int(X.shape[0]), "positive_fraction": float(y.mean())}

    if arm_cfg.get("smote"):
        from imblearn.over_sampling import SMOTE
        k = int(arm_cfg.get("smote_k_neighbors", 5))
        sm = SMOTE(random_state=int(cfg["seed"]), k_neighbors=k)
        X, y = sm.fit_resample(X, y)
    after = {"n": int(X.shape[0]), "positive_fraction": float(y.mean())}

    clf = make_classifier(arm_cfg.get("classifier", "hist_gb"), int(cfg["seed"]))
    started = time.time()
    clf.fit(X, y)
    elapsed = time.time() - started

    test_scenes = [s for s, sp in stats["splits"].items() if sp == "test"]
    sweep = ConfusionSweep()
    for scene in test_scenes:
        predict_test_scene(clf, cfg, scene,
                           out_dir / f"prediction_{scene}.tif", sweep)

    result = {
        "arm": arm,
        "model": "pixel_" + arm_cfg.get("classifier", "hist_gb"),
        "config_fingerprint": fingerprint(cfg),
        "features": list(FEATURE_NAMES),
        "smote": bool(arm_cfg.get("smote")),
        "train_pixels_before_resampling": before,
        "train_pixels_after_resampling": after,
        "fit_seconds": round(elapsed, 1),
        "test_scenes": test_scenes,
        **sweep.to_dict(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2),
                                          encoding="utf-8")
    b = result["best_f1"]
    print(f"  {arm:<22} AP={result['average_precision']:.3f}  "
          f"bestF1={b['f1']:.3f} @ t={b['threshold']:.2f}  "
          f"P={b['precision']:.3f} R={b['recall']:.3f} IoU={b['iou']:.3f}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="config/config.yaml", type=Path)
    ap.add_argument("--arms", default="config/arms.yaml", type=Path)
    ap.add_argument("--arm", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    arms = yaml.safe_load(args.arms.read_text(encoding="utf-8"))["pixel_arms"]
    stats = json.loads(
        (Path(cfg["processed_dir"]) / cfg["city"] / "stats.json")
        .read_text(encoding="utf-8"))

    chosen = list(arms) if args.all else (args.arm or [])
    if not chosen:
        raise SystemExit("pass --arm NAME (repeatable) or --all")
    unknown = [a for a in chosen if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; have {sorted(arms)}")

    for arm in chosen:
        print(f"\n=== {arm} ===")
        run_arm(arm, arms[arm], cfg, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
