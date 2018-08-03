import keras.backend as K
from tensorflow import confusion_matrix, diag_part


def _calc_classification_statistics(y_true, y_pred):
    """
    A shared template for computing each class's true-positive, true-negative, false-positive and false-negative.
    :param y_true: An array of the known labels, for all input samples.
    :param y_pred: An array of the predicted labels, for all input samples.
    :return: Four arrays of true-positive, true-negative, false-positive and false-negative for each class.
    """
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
    """
    A metric for computing the mean of recall among the possible classes.
    :param y_true: An array of the known labels, for all input samples.
    :param y_pred: An array of the predicted labels, for all input samples.
    :return: The mean of recalls.
    """
    true_positives, _, _, false_negatives = _calc_classification_statistics(y_true, y_pred)
    possible_positives = true_positives + false_negatives
    recall = true_positives / (possible_positives + K.epsilon())
    return K.mean(recall)


def _average_precision(y_true, y_pred):
    """
    A metric for computing the mean of precision among the possible classes.
    :param y_true: An array of the known labels, for all input samples.
    :param y_pred: An array of the predicted labels, for all input samples.
    :return: The mean of precisions
    """
    true_positives, false_positives, _, _ = _calc_classification_statistics(y_true, y_pred)
    predicted_positives = true_positives + false_positives
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# A dictionary for converting textual metrics representation the the metrics' methods.
_metric_name_to_func = {
    'average precision': _average_precision,
    'average recall': _average_recall,
    'total accuracy': 'accuracy'  # This one is automatically created by Keras.
}


def create_metrics(metrics_name):
    """
    A factory for metrics.
    :param metrics_name: A list of metrics' names
    :return: A list of the requested metrics methods.
    """
    metrics = []
    for metric_name in metrics_name:
        metric_name = metric_name.lower()
        if metric_name in _metric_name_to_func:
            metrics.append(_metric_name_to_func[metric_name])
        else:
            raise IOError('Metric {0} is NOT supported'.format(metric_name))
    return metrics
