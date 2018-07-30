import json
import datetime
from os import path
import boto3
import pandas as pd
import numpy as np

_logs_dir = path.join('logs')
s3_bucket_name = 'wfk'
s3_folder_path = 'EladEdenProject-MFML2018'


def read_experiments_config(experiments_config_path):
    with open(experiments_config_path, 'r') as experiments_file:
        experiments_data = json.load(experiments_file)
        return experiments_data


def save_model_weights(experiment_name, model):
    weights_file_name = experiment_name + '_' + str(datetime.date.today()) + '.h5'
    weights_file_name = weights_file_name.replace(' ', '_')
    weights_file_path = path.join(_logs_dir, 'ModelsWeights', weights_file_name)
    model.save_weights(filepath=weights_file_path)
    return weights_file_path


def upload_to_s3(tensorboard_logs, weight_files, experiments_logs):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(s3_bucket_name)

    for log in tensorboard_logs:
        bucket.upload_file(log, '{0}/TensoeboardLogs/{1}'.format(s3_folder_path, path.basename(log)))
    for weights in weight_files:
        bucket.upload_file(weights, '{0}/ModelsWeights/{1}'.format(s3_folder_path, path.basename(weights)))
    for experiment_log in experiments_logs:
        bucket.upload_file(experiment_log, '{0}/LayersOutput/{1}'.format(s3_folder_path, path.basename(experiment_log)))


# TODO: Complete using pandas tutorial.
def save_experiment_log():
    pass


def save_layers_logs(layers_data):
    for layer_index, layer_output in enumerate(layers_data):
        data_path = path.join(_logs_dir, 'LayersOutput', 'layer no {0}.txt'.format(layer_index))
        print(len(layer_output))
        print(layer_output.ndim)
        pd.DataFrame(layer_output.flatten()).to_csv(data_path)
