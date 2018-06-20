from tensorflow.python.keras.datasets import mnist, cifar10

_datasets_from_keras = {
    'mnist': mnist,
    'cifar10': cifar10
}


def create_dataset(dataset_name):
    dataset_as_lower = dataset_name.lower()
    if dataset_as_lower in _datasets_from_keras.keys():
        (x_train, y_train), (x_test, y_test) = _datasets_from_keras[dataset_as_lower].load_data()
    else:
        raise IOError("Dataset {0} is NOT supported".format(dataset_name))

    return x_train, y_train, x_test, y_test
