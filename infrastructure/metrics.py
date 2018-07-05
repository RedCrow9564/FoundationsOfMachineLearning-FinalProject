import keras.backend as K
from tensorflow import confusion_matrix, diag_part, divide, to_float
import numpy as np


# TODO: Add a metric for inner layers output.
def _calc_classification_statistics(y_true, y_pred):
    y_true = K.argmax(y_true, 1)
    y_pred = K.argmax(y_pred, 1)
    confuse_mat = confusion_matrix(y_true, y_pred, dtype='float32')
    tp = diag_part(confuse_mat)
    fp = K.sum(confuse_mat, axis=0) - tp
    fn = K.sum(confuse_mat, axis=1) - tp
    tn = K.sum(tp) - tp
    return tp, fp, tn, fn


# Note: Precision and Recall metrics only work when the output is encoded in one-hot notation
# (zeroes and ones vectors) and the loss function must NOT be sparse_categorical_cross_entropy.

def _average_recall(y_true, y_pred):
    true_positives, _, _, false_negatives = _calc_classification_statistics(y_true, y_pred)
    possible_positives = true_positives + false_negatives
    recall = true_positives / (possible_positives + K.epsilon())
    return K.mean(recall)


def _average_precision(y_true, y_pred):
    true_positives, false_positives, _, _ = _calc_classification_statistics(y_true, y_pred)
    predicted_positives = true_positives + false_positives
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


_metric_name_to_func = {
    'average precision': _average_precision,
    'average recall': _average_recall,
    'total accuracy': 'accuracy'  # This one is automatically created by Keras.
}


def create_metrics(metrics_name):
    metrics = []
    for metric_name in metrics_name:
        metric_name = metric_name.lower()
        if metric_name in _metric_name_to_func:
            metrics.append(_metric_name_to_func[metric_name])
        else:
            raise IOError('Metric {0} is NOT supported'.format(metric_name))
    return metrics
