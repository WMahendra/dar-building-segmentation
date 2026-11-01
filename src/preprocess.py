"""Rasterise building footprints and build a leakage-free tile index.

Three outputs, written into data/processed/<city>/:

  masks/<scene>.tif   binary building mask, on the imagery's own grid
  tiles.csv           one row per patch: scene, pixel window, positive fraction,
                      valid fraction, split
  stats.json          measured class balance and per-scene geometry

Why a tile *index* rather than exported patch arrays: the six Dar es Salaam
scenes are large COGs, and writing every 256x256 patch to disk would duplicate
them several times over once overlapping strides are used. The training
generator reads windows straight out of the COG instead, which is what COGs are
for.

Why the split is by scene and not by patch: neighbouring patches from one
flight overlap in content and share radiometry, so a random patch split would
leak the test set into training and inflate every metric. Whole scenes go to
one split only. With six scenes the default is 4 train / 1 val / 1 test, chosen
by scene id so the assignment is deterministic and reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window

from .raster import valid_mask_for, validity_rule


def rasterise_scene(image_path: Path, label_path: Path, mask_path: Path) -> dict:
    """Burn the footprints onto the imagery grid. Returns scene geometry info."""
    with rasterio.open(image_path) as src:
        profile = src.profile
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs

        gdf = gpd.read_file(label_path)
        if gdf.empty:
            shapes = []
        else:
            if gdf.crs is None:
                raise ValueError(f"{label_path} has no CRS; cannot align to imagery")
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)
            # Drop anything that is not a polygon: a stray point or line would
            # rasterise to nothing but still cost time.
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
            shapes = [(geom, 1) for geom in gdf.geometry]

        mask = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=1,
            dtype="uint8",
            all_touched=False,
        ) if shapes else np.zeros((height, width), dtype="uint8")

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8", nodata=None, compress="deflate",
                       tiled=True, blockxsize=256, blockysize=256)
    out_profile.pop("photometric", None)
    with rasterio.open(mask_path, "w", **out_profile) as dst:
        dst.write(mask, 1)

    return {
        "height": height,
        "width": width,
        "crs": str(crs),
        "n_footprints": int(len(shapes)),
        "building_pixel_fraction": float(mask.mean()),
    }


def index_scene(
    image_path: Path,
    mask_path: Path,
    scene: str,
    patch: int,
    stride: int,
    min_valid: float,
) -> list[dict]:
    rows: list[dict] = []
    with rasterio.open(image_path) as src, rasterio.open(mask_path) as msk:
        rule = validity_rule(src)
        for row_off in range(0, max(src.height - patch + 1, 1), stride):
            for col_off in range(0, max(src.width - patch + 1, 1), stride):
                win = Window(col_off, row_off, patch, patch)
                if col_off + patch > src.width or row_off + patch > src.height:
                    continue
                vm = valid_mask_for(src, win, rule)
                vf = float(vm.mean())
                if vf < min_valid:
                    continue
                m = msk.read(1, window=win)
                rows.append(
                    {
                        "scene": scene,
                        "row_off": row_off,
                        "col_off": col_off,
                        "patch": patch,
                        "valid_fraction": round(vf, 4),
                        "positive_fraction": round(float((m > 0).mean()), 4),
                    }
                )
    for r in rows:
        r["validity_rule"] = rule
    return rows


def assign_splits(scenes: list[str], n_val: int, n_test: int) -> dict[str, str]:
    """Deterministic scene-level split. Sorted order, not random, so the same
    scenes land in the same split on every machine and in every rerun."""
    ordered = sorted(scenes)
    if len(ordered) < n_val + n_test + 1:
        raise SystemExit(
            f"{len(ordered)} scene(s) cannot be split into "
            f"train/{n_val} val/{n_test} test"
        )
    split = {}
    for i, s in enumerate(ordered):
        if i < n_test:
            split[s] = "test"
        elif i < n_test + n_val:
            split[s] = "val"
        else:
            split[s] = "train"
    return split


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="data/processed", type=Path)
    ap.add_argument("--city", default="dar")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256,
                    help="256 = non-overlapping tiles; halve it to augment")
    ap.add_argument("--min-valid", type=float, default=0.90,
                    help="discard patches with more collar than this allows")
    ap.add_argument("--n-val", type=int, default=1)
    ap.add_argument("--n-test", type=int, default=1)
    args = ap.parse_args()

    img_dir = args.raw / args.city / "images"
    lbl_dir = args.raw / args.city / "labels"
    images = sorted(img_dir.glob("*.tif"))
    if not images:
        raise SystemExit(f"no imagery in {img_dir}; run src/fetch_data.py first")

    out_city = args.out / args.city
    stats: dict[str, dict] = {}
    all_rows: list[dict] = []

    for image_path in images:
        scene = image_path.stem
        label_path = lbl_dir / f"{scene}.geojson"
        if not label_path.exists():
            print(f"  ! {scene}: no label file, skipped")
            continue
        mask_path = out_city / "masks" / f"{scene}.tif"
        print(f"  {scene}: rasterising", flush=True)
        stats[scene] = rasterise_scene(image_path, label_path, mask_path)
        print(f"  {scene}: indexing tiles", flush=True)
        rows = index_scene(image_path, mask_path, scene,
                           args.patch, args.stride, args.min_valid)
        stats[scene]["n_tiles"] = len(rows)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("no usable tiles produced")

    split = assign_splits(list(stats), args.n_val, args.n_test)
    df = pd.DataFrame(all_rows)
    df["split"] = df["scene"].map(split)

    out_city.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_city / "tiles.csv", index=False)

    summary = {
        "city": args.city,
        "patch": args.patch,
        "stride": args.stride,
        "min_valid": args.min_valid,
        "splits": split,
        "scenes": stats,
        "tiles_per_split": df["split"].value_counts().to_dict(),
        # The single number the whole imbalance experiment turns on. It is
        # measured here, never assumed.
        "building_pixel_fraction_overall": float(
            df["positive_fraction"].mul(df["valid_fraction"]).sum()
            / df["valid_fraction"].sum()
        ),
        "building_pixel_fraction_by_split": {
            s: float(g["positive_fraction"].mul(g["valid_fraction"]).sum()
                     / g["valid_fraction"].sum())
            for s, g in df.groupby("split")
        },
        "empty_tile_fraction": float((df["positive_fraction"] == 0).mean()),
    }
    (out_city / "stats.json").write_text(json.dumps(summary, indent=2),
                                         encoding="utf-8")

    print(f"\ntiles        : {len(df)}  {summary['tiles_per_split']}")
    print(f"split        : {split}")
    print(f"building px  : {summary['building_pixel_fraction_overall']:.4f} "
          f"of valid pixels")
    print(f"empty tiles  : {summary['empty_tile_fraction']:.3f}")
    print(f"written      : {out_city/'tiles.csv'}, {out_city/'stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
