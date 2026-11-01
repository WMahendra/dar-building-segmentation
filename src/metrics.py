"""Threshold-swept confusion counts, and the metrics that matter here.

Overall accuracy is not reported anywhere in this project. On a class that
occupies a few percent of the pixels, predicting "no building" everywhere scores
in the mid-nineties, so accuracy would make the worst model look like the best.
What is reported instead: IoU and F1 on the building class, and precision and
recall separately, because the whole question of imbalance handling is which of
the two it trades away.

Counts are accumulated over a grid of thresholds in one pass. That keeps memory
flat regardless of how many tiles the test split holds, and it means the
precision-recall curve and the best-threshold choice come out of the same
numbers as the headline metrics rather than a separate run.
"""

from __future__ import annotations

import numpy as np

DEFAULT_THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)


class ConfusionSweep:
    """TP/FP/FN/TN per threshold, accumulated batch by batch."""

    def __init__(self, thresholds: np.ndarray = DEFAULT_THRESHOLDS):
        self.thresholds = np.asarray(thresholds, dtype="float32")
        n = len(self.thresholds)
        self.tp = np.zeros(n, dtype="int64")
        self.fp = np.zeros(n, dtype="int64")
        self.fn = np.zeros(n, dtype="int64")
        self.tn = np.zeros(n, dtype="int64")

    def update(self, y_true: np.ndarray, y_prob: np.ndarray,
               valid: np.ndarray | None = None) -> None:
        t = (np.asarray(y_true).ravel() > 0.5)
        p = np.asarray(y_prob).ravel()
        if valid is not None:
            v = np.asarray(valid).ravel().astype(bool)
            t, p = t[v], p[v]
        pos = t.sum()
        neg = t.size - pos
        for i, th in enumerate(self.thresholds):
            pred = p >= th
            tp = np.count_nonzero(pred & t)
            fp = np.count_nonzero(pred) - tp
            self.tp[i] += tp
            self.fp[i] += fp
            self.fn[i] += pos - tp
            self.tn[i] += neg - fp

    # -- derived ----------------------------------------------------------
    def table(self) -> dict[str, np.ndarray]:
        tp, fp, fn = self.tp.astype(float), self.fp.astype(float), self.fn.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f1 = np.where(precision + recall > 0,
                          2 * precision * recall / (precision + recall), 0.0)
            iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        return {
            "threshold": self.thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }

    def best(self, by: str = "f1") -> dict[str, float]:
        t = self.table()
        i = int(np.nanargmax(t[by]))
        return {
            "threshold": float(t["threshold"][i]),
            "precision": float(t["precision"][i]),
            "recall": float(t["recall"][i]),
            "f1": float(t["f1"][i]),
            "iou": float(t["iou"][i]),
            "tp": int(t["tp"][i]), "fp": int(t["fp"][i]),
            "fn": int(t["fn"][i]), "tn": int(t["tn"][i]),
        }

    def at(self, threshold: float = 0.5) -> dict[str, float]:
        t = self.table()
        i = int(np.argmin(np.abs(t["threshold"] - threshold)))
        return {
            "threshold": float(t["threshold"][i]),
            "precision": float(t["precision"][i]),
            "recall": float(t["recall"][i]),
            "f1": float(t["f1"][i]),
            "iou": float(t["iou"][i]),
        }

    def average_precision(self) -> float:
        """Area under the precision-recall curve, by the step rule.

        Computed from the swept counts, so it is an approximation on the
        threshold grid rather than the exact all-thresholds value. Reported as a
        threshold-free summary: it is the one number that compares arms without
        first committing to an operating point.
        """
        t = self.table()
        order = np.argsort(t["recall"])
        r = np.concatenate([[0.0], t["recall"][order]])
        p = np.concatenate([[t["precision"][order][0]], t["precision"][order]])
        return float(np.sum(np.diff(r) * p[1:]))

    def to_dict(self) -> dict:
        t = self.table()
        return {
            "sweep": {k: np.asarray(v).tolist() for k, v in t.items()},
            "at_0.5": self.at(0.5),
            "best_f1": self.best("f1"),
            "average_precision": self.average_precision(),
            "positive_pixel_fraction": float(
                (self.tp[0] + self.fn[0])
                / max(self.tp[0] + self.fn[0] + self.fp[0] + self.tn[0], 1)
            ),
        }
