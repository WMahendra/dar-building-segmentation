"""Raster helpers shared by preprocessing, training and evaluation.

Kept separate from data.py so that preprocessing does not have to import
TensorFlow: rasterising labels on a laptop should not need a deep-learning
runtime, and the import alone costs several seconds and a GPU probe.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.windows import Window


def validity_rule(src: rasterio.DatasetReader) -> str:
    """Decide once, per dataset, how valid pixels are identified.

    Drone mosaics are padded where the flight lines do not cover the bounding
    box, and those pixels must be excluded from tiling and from metrics, or the
    background class is inflated by sensor geometry rather than by what is on the
    ground. How the padding is marked, though, varies by how the mosaic was
    written, so the rule is chosen from the file's own metadata:

      "alpha"    the file declares an alpha band (GDAL mask flag, or a band
                 tagged alpha). Authoritative.
      "nodata"   an explicit nodata value is set. Authoritative.
      "binary4"  four bands, no declared alpha, and the fourth holds only two
                 values. Almost certainly an undeclared alpha band: some COG
                 writers set photometric=RGB, which leaves GDAL no slot to mark
                 the fourth band with, so it reports "undefined".
      "allzero"  last resort - a pixel that is zero in every band is treated as
                 padding. This is the only rule that can misfire on real data,
                 because a genuinely black roof is valid, which is why it comes
                 last and why preprocess.py prints which rule fired.

    Returning the rule as a string rather than silently applying it is the point:
    an incorrect choice changes every reported metric, so it belongs in the run
    log where it can be checked against the imagery.
    """
    flags = src.mask_flag_enums
    if any(MaskFlags.alpha in f for f in flags):
        return "alpha"
    if src.colorinterp and ColorInterp.alpha in src.colorinterp:
        return "alpha"
    if src.nodata is not None:
        return "nodata"
    if src.count == 4:
        sample = src.read(4, out_shape=(min(src.height, 512), min(src.width, 512)))
        if np.unique(sample).size <= 2:
            return "binary4"
    return "allzero"


def valid_mask_for(src: rasterio.DatasetReader, window: Window,
                   rule: str | None = None) -> np.ndarray:
    """True where the pixel carries real data, under the given rule.

    Pass the rule from validity_rule() to avoid re-deciding per window; omitting
    it is convenient but costs a metadata inspection on every call.
    """
    rule = rule or validity_rule(src)

    if rule == "alpha":
        # GDAL's own notion of validity, which resolves alpha and internal masks
        # without us having to find the band ourselves.
        return src.dataset_mask(window=window) > 0
    if rule == "nodata":
        arr = src.read(window=window)
        return np.any(arr != src.nodata, axis=0)
    if rule == "binary4":
        band = src.read(4, window=window)
        return band > band.min()
    arr = src.read(window=window)
    return np.any(arr != 0, axis=0)


def scale_to_unit(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Map integer imagery to [0, 1] using the dtype maximum.

    Dividing by the dtype maximum needs no statistics, so the same scaling
    applies identically on every machine and to every scene. A percentile
    stretch would adapt to each scene's own histogram, which is exactly what we
    do not want when the test scene is there to measure transfer to unseen
    radiometry.
    """
    out = arr.astype("float32")
    if dtype == "uint8":
        return out / 255.0
    return out / float(np.iinfo(dtype).max)
