"""Train one arm of the imbalance experiment and evaluate it on the test scene.

One invocation = one arm. Everything that is not the treatment is pinned: the
same U-Net, the same split, the same seed, the same epoch budget and the same
optimiser settings. Only the loss function and the sampling fraction change, so
a difference in the results table is attributable.

    python -m src.train --arm bce
    python -m src.train --arm weighted_bce
    python -m src.train --all          # every arm in config/arms.yaml, in turn

Each arm writes results/<arm>/: weights, history.json, metrics.json and the
stitched probability raster for the test scene.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window

from . import losses as loss_zoo
from .data import sequences
from .metrics import ConfusionSweep
from .provenance import fingerprint
from .models import unet
from .raster import scale_to_unit, valid_mask_for, validity_rule


def set_seeds(seed: int) -> None:
    """Pin every RNG the run touches.

    Full bit-for-bit determinism on a GPU would also need
    TF_DETERMINISTIC_OPS=1, which costs speed; the aim here is that arms start
    from the same initialisation and see the same tile order, which these seeds
    do guarantee.
    """
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_stats(processed: Path, city: str) -> dict:
    p = Path(processed) / city / "stats.json"
    if not p.exists():
        raise SystemExit(f"{p} not found; run src/preprocess.py first")
    return json.loads(p.read_text(encoding="utf-8"))


def predict_test_scene(model, cfg: dict, scene: str, out_tif: Path,
                       sweep: ConfusionSweep) -> None:
    """Run the model over the whole test scene and stitch the probabilities.

    Tiles are predicted on the same non-overlapping grid preprocess.py indexed,
    then written into a single-band raster that carries the imagery's CRS and
    transform, so the output opens on top of the source in QGIS. Confusion
    counts are accumulated as we go, over valid pixels only, so the collar
    around the mosaic cannot contribute to the metrics.
    """
    image = Path(cfg["raw_dir"]) / cfg["city"] / "images" / f"{scene}.tif"
    mask = Path(cfg["processed_dir"]) / cfg["city"] / "masks" / f"{scene}.tif"
    patch = int(cfg["patch"])
    bands = int(cfg["bands"])

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
                batch, windows = [], []
                for col in range(0, src.width - patch + 1, patch):
                    win = Window(col, row, patch, patch)
                    x = src.read(indexes=list(range(1, bands + 1)), window=win)
                    batch.append(scale_to_unit(np.transpose(x, (1, 2, 0)),
                                               src.dtypes[0]))
                    windows.append(win)
                if not batch:
                    continue
                probs = model.predict(np.stack(batch), verbose=0)[..., 0]
                for p, win in zip(probs, windows):
                    v = valid_mask_for(src, win, rule)
                    y = msk.read(1, window=win)
                    sweep.update(y, p, valid=v)
                    q = np.where(v, np.round(p * 254).astype("uint8"), 255)
                    dst.write(q, 1, window=win)


def run_arm(arm: str, arm_cfg: dict, cfg: dict, stats: dict) -> dict:
    import tensorflow as tf

    set_seeds(int(cfg["seed"]))
    out_dir = Path(cfg["results_dir"]) / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    f = float(stats["building_pixel_fraction_overall"])
    if not 0.0 < f < 1.0:
        raise SystemExit(f"measured positive fraction {f} is degenerate")

    kwargs = dict(arm_cfg.get("loss_kwargs") or {})
    if arm_cfg["loss"] == "weighted_bce" and "pos_weight" not in kwargs:
        kwargs["pos_weight"] = (1.0 - f) / f  # inverse class frequency, measured
    loss_fn = loss_zoo.build(arm_cfg["loss"], **kwargs)

    seqs = sequences(
        cfg["processed_dir"], cfg["raw_dir"], cfg["city"],
        batch_size=int(cfg["batch_size"]), bands=int(cfg["bands"]),
        oversample_positive=arm_cfg.get("oversample_positive"),
        seed=int(cfg["seed"]),
    )
    if "train" not in seqs or "val" not in seqs:
        raise SystemExit("train and val splits are both required")

    model = unet(
        input_shape=(int(cfg["patch"]), int(cfg["patch"]), int(cfg["bands"])),
        depth=int(cfg["depth"]), base_filters=int(cfg["base_filters"]),
        dropout=float(cfg["dropout"]),
        output_bias=math.log(f / (1.0 - f)),
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(cfg["learning_rate"])),
        loss=loss_fn,
    )

    started = time.time()
    history = model.fit(
        seqs["train"],
        validation_data=seqs["val"],
        epochs=int(cfg["epochs"]),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(out_dir / "weights.keras"), monitor="val_loss",
                save_best_only=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=int(cfg["patience"]),
                restore_best_weights=True),
        ],
        verbose=int(cfg.get("verbose", 1)),
    )
    elapsed = time.time() - started

    (out_dir / "history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.history.items()},
                   indent=2), encoding="utf-8")

    # Evaluate on the held-out scene, which the model has never seen.
    test_scenes = [s for s, sp in stats["splits"].items() if sp == "test"]
    sweep = ConfusionSweep()
    for scene in test_scenes:
        predict_test_scene(model, cfg, scene,
                           out_dir / f"prediction_{scene}.tif", sweep)

    result = {
        "arm": arm,
        "loss": arm_cfg["loss"],
        "config_fingerprint": fingerprint(cfg),
        "loss_kwargs": {k: float(v) for k, v in kwargs.items()},
        "oversample_positive": arm_cfg.get("oversample_positive"),
        "epochs_run": len(history.history["loss"]),
        "train_seconds": round(elapsed, 1),
        "test_scenes": test_scenes,
        "measured_positive_fraction": f,
        **sweep.to_dict(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2),
                                          encoding="utf-8")
    for s in seqs.values():
        s.close()

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
    arms = yaml.safe_load(args.arms.read_text(encoding="utf-8"))["arms"]
    stats = load_stats(cfg["processed_dir"], cfg["city"])

    if args.all:
        chosen = list(arms)
    elif args.arm:
        missing = [a for a in args.arm if a not in arms]
        if missing:
            raise SystemExit(f"unknown arm(s) {missing}; have {sorted(arms)}")
        chosen = args.arm
    else:
        raise SystemExit("pass --arm NAME (repeatable) or --all")

    print(f"city {cfg['city']} | positive fraction "
          f"{stats['building_pixel_fraction_overall']:.4f} | "
          f"splits {stats['splits']}")
    for arm in chosen:
        print(f"\n=== {arm} ===")
        run_arm(arm, arms[arm], cfg, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
