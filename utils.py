import json
import datetime
import os

_logs_dir = os.path.join('logs')
_experiments_input_file = 'experiments_config.json'


def read_experiments_config():
    with open(_experiments_input_file, 'r') as experiments_file:
        experiments_data = json.load(experiments_file)
        return experiments_data


# TODO: Complete using Kerad FAQs
def read_weights_file(weights_file_path):
    pass


def save_model_weights(experiment_name, model):
    weights_file_name = experiment_name + '_' + str(datetime.date.today()) + '.h5'
    weights_file_name.replace(' ', '_')
    weights_file_path = os.path.join(_logs_dir, 'ModelsWeights', weights_file_name)
    model.save_weights(filepath=weights_file_path)


# TODO: Complete using boto3 library
def upload_to_s3():
    pass


# TODO: Complete using pandas tutorial.
def save_experiment_log():
    pass

# TODO: Complete using pandas tutorial.
def save_layers_logs():
    pass
