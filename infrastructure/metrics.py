import keras.backend as K
from sklearn.metrics import confusion_matrix
import numpy as np


# TODO: Add a metric for inner layers output.
def _calc_classification_statistics(y_true, y_pred):
    confuse_mat = confusion_matrix(y_true, y_pred)
    tp = np.diag(confuse_mat)
    fp = np.sum(confuse_mat, axis=0) - tp
    fn = np.sum(confuse_mat, axis=1) - tp
    tn = np.sum(tp) - tp
    return tp, fp, tn, fn


# Note: Precision and Recall metrics only work when the output is encoded in one-hot notation
# (zeroes and ones vectors) and the loss function must NOT be sparse_categorical_cross_entropy.

def _average_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def _average_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
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
