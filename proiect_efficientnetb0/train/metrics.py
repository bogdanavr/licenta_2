from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], y_prob=None):
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        top2 = np.argsort(y_prob, axis=1)[:, -2:]
        metrics['top2_accuracy'] = float(np.mean([label in preds for label, preds in zip(y_true, top2)]))
        try:
            metrics['macro_auc_ovr'] = float(
                roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            )
        except ValueError:
            metrics['macro_auc_ovr'] = None
    else:
        metrics['top2_accuracy'] = None
        metrics['macro_auc_ovr'] = None

    return metrics
