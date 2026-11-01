"""Generate synthetic scenes shaped exactly like the Open Cities data.

The real imagery cannot be downloaded in every environment (a sandbox without
egress, CI, a laptop on a train), and a pipeline that is only ever exercised
against the real 500 MB download is a pipeline whose bugs are found on the GPU
run. These fixtures match what matters structurally: uint8 RGB + alpha COG-ish
GeoTIFFs with a projected CRS, a transparent collar, polygon footprints in a
GeoJSON with a CRS, and a deliberately low positive fraction.

They are not a substitute for results. Metrics computed on them mean nothing
about building detection; they only prove the code paths run, shapes line up,
georeferencing survives and the arms produce distinguishable output.

    python -m tests.make_synthetic --out data/raw --city dar --scenes 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def one_scene(rng: np.random.Generator, side: int, n_buildings: int):
    """Return (rgba uint8 array (4, side, side), list of polygon pixel boxes)."""
    # Speckled ground, so local texture features have something to find.
    ground = rng.normal(120, 18, size=(side, side, 3)).clip(0, 255)
    rgba = np.zeros((4, side, side), dtype="uint8")
    boxes = []

    for _ in range(n_buildings):
        h = int(rng.integers(12, 34))
        w = int(rng.integers(12, 34))
        r0 = int(rng.integers(0, side - h))
        c0 = int(rng.integers(0, side - w))
        # Uniform bright roof: distinguishable by low local variance, which is
        # the kind of cue the pixel baseline is meant to be able to use.
        tone = rng.integers(170, 240)
        ground[r0:r0 + h, c0:c0 + w, :] = tone + rng.normal(0, 4, (h, w, 3))
        boxes.append((r0, c0, h, w))

    rgba[:3] = np.transpose(ground.clip(0, 255), (2, 0, 1)).astype("uint8")
    rgba[3] = 255
    # Collar on two edges, as drone mosaics have: both the alpha band and the
    # colour bands go to zero there, which is what the real writers produce.
    collar = max(side // 16, 1)
    rgba[:, :collar, :] = 0
    rgba[:, :, -collar:] = 0
    return rgba, boxes


def write_scene(out_img: Path, out_lbl: Path, rgba: np.ndarray,
                boxes, origin=(500000.0, 9250000.0), res: float = 0.07):
    side = rgba.shape[1]
    transform = from_origin(origin[0], origin[1], res, res)
    crs = "EPSG:32737"  # UTM 37S, which is where Dar es Salaam actually falls

    out_img.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_img, "w", driver="GTiff", height=side, width=side, count=4,
        dtype="uint8", crs=crs, transform=transform, compress="deflate",
        tiled=True, blockxsize=256, blockysize=256,
        # No photometric="rgb" here: with four bands it leaves GDAL no slot to
        # tag the fourth, which is exactly the "undefined alpha" case the
        # validity_rule fallbacks exist for. Declaring it properly means the
        # fixture exercises the primary path.
    ) as dst:
        dst.write(rgba)
        dst.colorinterp = [
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        ]

    features = []
    for (r0, c0, h, w) in boxes:
        xs, ys = [], []
        for (r, c) in ((r0, c0), (r0, c0 + w), (r0 + h, c0 + w), (r0 + h, c0)):
            x, y = transform * (c, r)
            xs.append(x)
            ys.append(y)
        ring = [[xs[i], ys[i]] for i in range(4)] + [[xs[0], ys[0]]]
        features.append({
            "type": "Feature",
            "properties": {"building": "yes"},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    out_lbl.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32737"}},
        "features": features,
    }), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/raw", type=Path)
    ap.add_argument("--city", default="dar")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--side", type=int, default=1024)
    ap.add_argument("--buildings", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Hex-like ids, so the fixtures sort and read like the real scene names.
    ids = [f"{rng.integers(0x100000, 0xFFFFFF):06x}" for _ in range(args.scenes)]

    for i, scene in enumerate(ids):
        rgba, boxes = one_scene(rng, args.side, args.buildings)
        write_scene(
            args.out / args.city / "images" / f"{scene}.tif",
            args.out / args.city / "labels" / f"{scene}.geojson",
            rgba, boxes,
            origin=(500000.0 + i * 1000.0, 9250000.0),
        )
        print(f"  {scene}: {args.side}x{args.side}, {len(boxes)} footprints")

    print(f"\nwrote {args.scenes} synthetic scenes to {args.out/args.city}")
    print("These are fixtures for exercising the code. Any metric computed on "
          "them is meaningless as a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
