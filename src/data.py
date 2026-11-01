"""Tile loading for Keras, reading windows straight out of the COGs.

Patches are never materialised on disk (see preprocess.py). Each batch opens the
scene's COG at the recorded window, which is cheap because the rasters are
tiled. Dataset handles are cached per worker, since reopening a COG for every
patch would dominate the step time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tensorflow import keras

from .raster import scale_to_unit, validity_rule


class TileSequence(keras.utils.Sequence):
    """Batches of (image, mask) patches for one split.

    Parameters
    ----------
    augment
        Horizontal/vertical flips and 90-degree rotations only. These are the
        transforms that are safe for nadir aerial imagery: a building is still a
        building upside down, whereas a brightness or colour jitter would blur
        the radiometric differences between scenes that the cross-scene test set
        exists to measure.
    oversample_positive
        Fraction of each batch drawn from tiles that contain at least one
        building pixel. None keeps the natural tile distribution. This is the
        *sampling* lever in the imbalance experiment; the loss functions are the
        other lever, and the two are varied independently.
    """

    def __init__(
        self,
        tiles: pd.DataFrame,
        image_dir: Path,
        mask_dir: Path,
        batch_size: int = 16,
        bands: int = 3,
        augment: bool = False,
        oversample_positive: float | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.tiles = tiles.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.batch_size = batch_size
        self.bands = bands
        self.augment = augment
        self.oversample_positive = oversample_positive
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._handles: dict[str, tuple] = {}

        self._positive = self.tiles.index[self.tiles["positive_fraction"] > 0].to_numpy()
        self._negative = self.tiles.index[self.tiles["positive_fraction"] == 0].to_numpy()
        if oversample_positive is not None and len(self._positive) == 0:
            raise ValueError("oversample_positive requested but no positive tiles")

        self.order = self.tiles.index.to_numpy()
        self.on_epoch_end()

    # -- plumbing ---------------------------------------------------------
    def _open(self, scene: str):
        if scene not in self._handles:
            self._handles[scene] = (
                rasterio.open(self.image_dir / f"{scene}.tif"),
                rasterio.open(self.mask_dir / f"{scene}.tif"),
            )
        return self._handles[scene]

    def __len__(self) -> int:
        return int(np.ceil(len(self.order) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.oversample_positive is not None:
            n = len(self.tiles)
            n_pos = int(round(n * self.oversample_positive))
            n_neg = n - n_pos
            pos = self.rng.choice(self._positive, size=n_pos,
                                  replace=n_pos > len(self._positive))
            neg = (self.rng.choice(self._negative, size=n_neg,
                                   replace=n_neg > len(self._negative))
                   if len(self._negative) else np.empty(0, dtype=int))
            self.order = np.concatenate([pos, neg]).astype(int)
            self.rng.shuffle(self.order)
        elif self.shuffle:
            self.order = self.rng.permutation(self.tiles.index.to_numpy())

    # -- reading ----------------------------------------------------------
    def _read(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        r = self.tiles.loc[i]
        src, msk = self._open(r["scene"])
        win = Window(int(r["col_off"]), int(r["row_off"]),
                     int(r["patch"]), int(r["patch"]))
        x = src.read(indexes=list(range(1, self.bands + 1)), window=win)
        x = scale_to_unit(np.transpose(x, (1, 2, 0)), src.dtypes[0])
        y = msk.read(1, window=win).astype("float32")[..., None]
        return x, y

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = int(self.rng.integers(4))
        if k:
            x, y = np.rot90(x, k, (0, 1)), np.rot90(y, k, (0, 1))
        if self.rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        if self.rng.random() < 0.5:
            x, y = x[::-1], y[::-1]
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    def __getitem__(self, b: int):
        idx = self.order[b * self.batch_size : (b + 1) * self.batch_size]
        xs, ys = [], []
        for i in idx:
            x, y = self._read(int(i))
            if self.augment:
                x, y = self._augment(x, y)
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)

    def close(self) -> None:
        for a, b in self._handles.values():
            a.close()
            b.close()
        self._handles.clear()


def load_tiles(processed_dir: Path, city: str) -> pd.DataFrame:
    path = Path(processed_dir) / city / "tiles.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run src/preprocess.py first")
    return pd.read_csv(path)


def sequences(
    processed_dir: Path,
    raw_dir: Path,
    city: str,
    batch_size: int = 16,
    bands: int = 3,
    oversample_positive: float | None = None,
    seed: int = 0,
) -> dict[str, TileSequence]:
    tiles = load_tiles(processed_dir, city)
    image_dir = Path(raw_dir) / city / "images"
    mask_dir = Path(processed_dir) / city / "masks"
    out = {}
    for split in ("train", "val", "test"):
        part = tiles[tiles["split"] == split]
        if part.empty:
            continue
        out[split] = TileSequence(
            part,
            image_dir,
            mask_dir,
            batch_size=batch_size,
            bands=bands,
            augment=(split == "train"),
            oversample_positive=oversample_positive if split == "train" else None,
            shuffle=(split == "train"),
            seed=seed,
        )
    return out
