import json
import datetime
from os import path
import boto3
import pandas as pd
from keras.callbacks import TensorBoard
import tensorflow as tf

_logs_dir = 'logs'
weights_files_path = path.join(_logs_dir, 'ModelsWeights')
experiment_logs_path = path.join(_logs_dir, 'ExperimentLogs')
layers_logs_path = path.join(_logs_dir, 'LayersOutput')
tensorboard_logs_path = path.join(_logs_dir, 'TensorBoard')
s3_bucket_name = 'wfk'
s3_folder_path = 'EladEdenProject-MFML2018'


# TODO: Complete commenting this file.
def read_experiments_config(experiments_config_path):
    """
    A for opening the JSON config file for the performed experiments.

    :param experiments_config_path: A path (relative or full) to the config JSON file.
    :return: A dictionary of all values in the file.
    """
    with open(experiments_config_path, 'r') as experiments_file:
        experiments_data = json.load(experiments_file)
        return experiments_data


def save_model_weights(experiment_name, model):
    """
    A function for saving a model's weights to a file.

    :param experiment_name: The textual name of this experiment.
    :param model: The model object.
    :return: A relative path to the new weights file.
    """
    weights_file_name = experiment_name + '_' + str(datetime.date.today()) + '.h5'
    weights_file_name = weights_file_name.replace(' ', '_')
    weights_file_path = path.join(weights_files_path, weights_file_name)
    model.save_weights(filepath=weights_file_path)
    return weights_file_path


def upload_to_s3(tensorboard_logs, experiment_results, weight_files, experiments_logs):
    credentials = read_experiments_config('credentials.json')
    session = boto3.Session(
        aws_access_key_id=credentials["aws-access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket(s3_bucket_name)

    for log in tensorboard_logs:
        bucket.upload_file(log, '{0}/TensorboardLogs/{1}'.format(s3_folder_path, path.basename(log)))
    for log in experiment_results:
        bucket.upload_file(log, '{0}/ExperimentsLogs/{1}'.format(s3_folder_path, path.basename(log)))
    for weights in weight_files:
        bucket.upload_file(weights, '{0}/ModelsWeights/{1}'.format(s3_folder_path, path.basename(weights)))
    for experiment_log in experiments_logs:
        s3_path = path.join('{0}', 'LayersOutput', '{1}').format(s3_folder_path, path.basename(experiment_log))
        local_path = path.join(_logs_dir, 'LayersOutput', '{0}').format(path.basename(experiment_log))
        bucket.upload_file(local_path, s3_path)


def save_experiment_log(results, experiment_name):
    """
    A function for saving the experiments results to a file.

    :param results: The results dictionary collected in the experiment.
    :param experiment_name: The experiment textual name.
    :return: The results file name.
    """
    data_not_to_save = ['Layers Testing Output', 'Layers Training Output']
    data_to_save = {k: v for k, v in results.items() if k not in data_not_to_save}
    file_name = path.join(experiment_logs_path, '{0} results.json'.format(experiment_name))
    with open(file_name, 'w') as outfile:
        outfile.write(json.dumps(data_to_save, indent=4))
    return file_name


def save_layers_logs(layers_data, data_name):
    """
    A function for saving inner layers outputs to a file.
    :param layers_data: The list of all layers outputs.
    :param data_name: A textual description of the data, used in the file's name.
    :return: A list of all files names that were created.
    """
    all_files = []
    for layer_index, layer_output in enumerate(layers_data):
        file_name = '{0} layer no {1}.txt'.format(data_name, layer_index)
        data_path = path.join(layers_logs_path, file_name)
        print(len(layer_output))
        print(layer_output.ndim)
        all_files.append(file_name)
        pd.DataFrame(layer_output.flatten()).to_csv(data_path, index=False)
    return all_files


class TrainValTensorBoard(TensorBoard):
    """
    A class for a TensorBoard logger which places training metrics and validation metrics in the same graphs.
    I do NOT own this class, it was posted by "Yu-Yang" at "https://stackoverflow.com/questions/47877475/
    keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure/48393723#48393723"

    """
    def __init__(self, log_dir, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = path.join(log_dir, 'validation')
        self.val_writer = None

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
