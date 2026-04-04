"""Segmentation metrics — mIoU, pixel accuracy, F1."""
from __future__ import annotations

import numpy as np


class RunningScore:
    def __init__(self, n_classes: int, ignore_index: int | None = None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        if ignore_index is not None and 0 <= ignore_index < n_classes:
            self.ignore_index = (ignore_index,)
        else:
            self.ignore_index = None

    def _fast_hist(self, label_true, label_pred):
        n = self.n_classes
        mask = (label_true >= 0) & (label_true < n)
        hist = np.bincount(
            n * label_true[mask].astype(int) + label_pred[mask],
            minlength=n ** 2,
        ).reshape(n, n)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def get_scores(self) -> dict:
        hist = self.confusion_matrix.copy()
        if self.ignore_index is not None:
            for idx in self.ignore_index:
                hist = np.delete(hist, idx, axis=0)
                hist = np.delete(hist, idx, axis=1)

        acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
        mean_iou = np.nanmean(iu)
        freq = hist.sum(axis=1) / (hist.sum() + 1e-10)
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        return {
            "pixel_acc": float(acc),
            "mIoU": float(mean_iou),
            "fwIoU": float(fw_iou),
            "class_iou": iu.tolist(),
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
