import tensorflow.python.keras.backend as K


# TODO: Elaborate metrics for multi-class cases
# TODO: Add a metric for inner layers output.
def _precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def _recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


_metric_name_to_func = {
    'precision': _precision,
    'recall': _recall,
    'accuracy': 'accuracy'  # This one is automatically created by Keras.
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
