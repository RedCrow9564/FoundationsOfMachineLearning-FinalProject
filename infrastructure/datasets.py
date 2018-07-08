from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
from keras import backend as K

_datasets_from_keras = {
    'mnist': {
        'data': mnist,
        'data type': 'image',
        'sample size': (28, 28),
        'channels': 1,
        'bits per sample': 8,
        'classes count': 10
    },
    'cifar10': {
        'data': cifar10,
        'data type': 'image',
        'sample size': (32, 32),
        'channels': 3,
        'bits per sample': 8,
        'classes count': 10
    }
}


def _pre_process_images(images, details):
    # If the images are gray-scale, the number of channels (1) must be "added" to the size of the samples.
    if details['channels'] == 1:
        img_rows, img_cols = details['sample size']

        # The place of the dimension with 1 depends on the backend used by Keras.
        if K.image_data_format() == 'channels_first':
            images = images.reshape(images.shape[0], 1, img_rows, img_cols)
        else:
            images = images.reshape(images.shape[0], img_rows, img_cols, 1)

    # Normalize pixel values to be in the interval [0, 1]
    images = images.astype('float32')
    max_bit_value = 2 ** details['bits per sample'] - 1
    images /= max_bit_value
    return images


def create_dataset(dataset_name):
    dataset_as_lower = dataset_name.lower()
    if dataset_as_lower in _datasets_from_keras.keys():
        data_details = _datasets_from_keras[dataset_as_lower]
        (x_train, y_train), (x_test, y_test) = data_details['data'].load_data()

        if data_details['data type'] == 'image':
            x_train = _pre_process_images(x_train, data_details)
            x_test = _pre_process_images(x_test, data_details)
    else:
        raise IOError("Dataset {0} is NOT supported".format(dataset_name))

    return x_train, y_train, x_test, y_test
