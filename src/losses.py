"""Loss functions for binary segmentation under class imbalance.

These are the treatment arms. Each addresses the imbalance differently, and the
point of the experiment is that they are not interchangeable:

  bce               plain binary cross-entropy. The control. Minimises average
                    pixel error, which a model can do well at by under-predicting
                    the minority class.
  weighted_bce      cross-entropy with the positive class upweighted. Shifts the
                    decision threshold implicitly; recall rises, precision falls.
  dice              optimises overlap directly, so it is insensitive to the
                    number of background pixels. No weight to tune.
  tversky           Dice with asymmetric penalties: beta > 0.5 punishes false
                    negatives harder than false positives.
  focal             down-weights easy pixels so the gradient concentrates on the
                    hard ones, which under imbalance are mostly the minority class.
  bce_dice          sum of the control and the overlap term; often more stable
                    early in training than Dice alone, whose gradient is small
                    when predictions start near zero.

All take y_true, y_pred with shape (batch, H, W, 1) and y_pred already in [0, 1].
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

EPS = keras.backend.epsilon()


def _flat(y_true, y_pred):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    return y_true, tf.clip_by_value(y_pred, EPS, 1.0 - EPS)


def bce(y_true, y_pred):
    y_true, y_pred = _flat(y_true, y_pred)
    return -tf.reduce_mean(
        y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    )


def weighted_bce(pos_weight: float):
    """Cross-entropy with the positive class scaled by pos_weight.

    A principled starting value is the inverse class frequency measured by
    preprocess.py (stats.json -> building_pixel_fraction_overall), i.e.
    (1 - f) / f. train.py uses that unless a value is given explicitly, so the
    weight comes from the data rather than from a guess.
    """

    def loss(y_true, y_pred):
        y_true, y_pred = _flat(y_true, y_pred)
        return -tf.reduce_mean(
            pos_weight * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

    loss.__name__ = f"weighted_bce_{pos_weight:g}"
    return loss


def dice(smooth: float = 1.0):
    def loss(y_true, y_pred):
        y_true, y_pred = _flat(y_true, y_pred)
        inter = tf.reduce_sum(y_true * y_pred)
        total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        return 1.0 - (2.0 * inter + smooth) / (total + smooth)

    loss.__name__ = "dice"
    return loss


def tversky(beta: float = 0.7, smooth: float = 1.0):
    """beta is the weight on false negatives; beta = 0.5 reduces to Dice."""

    def loss(y_true, y_pred):
        y_true, y_pred = _flat(y_true, y_pred)
        tp = tf.reduce_sum(y_true * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        return 1.0 - (tp + smooth) / (
            tp + beta * fn + (1.0 - beta) * fp + smooth
        )

    loss.__name__ = f"tversky_{beta:g}"
    return loss


def focal(gamma: float = 2.0, alpha: float = 0.25):
    def loss(y_true, y_pred):
        y_true, y_pred = _flat(y_true, y_pred)
        pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        a = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return -tf.reduce_mean(a * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))

    loss.__name__ = f"focal_g{gamma:g}_a{alpha:g}"
    return loss


def bce_dice(smooth: float = 1.0):
    d = dice(smooth)

    def loss(y_true, y_pred):
        return bce(y_true, y_pred) + d(y_true, y_pred)

    loss.__name__ = "bce_dice"
    return loss


def build(name: str, pos_weight: float | None = None, **kw):
    """Resolve a loss by name, as written in config.yaml."""
    name = name.lower()
    if name == "bce":
        return bce
    if name == "weighted_bce":
        if pos_weight is None:
            raise ValueError("weighted_bce needs pos_weight")
        return weighted_bce(pos_weight)
    if name == "dice":
        return dice(**kw)
    if name == "tversky":
        return tversky(**kw)
    if name == "focal":
        return focal(**kw)
    if name == "bce_dice":
        return bce_dice(**kw)
    raise ValueError(f"unknown loss {name!r}")
