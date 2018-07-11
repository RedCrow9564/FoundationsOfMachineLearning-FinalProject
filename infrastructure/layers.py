from keras.layers import Layer, Activation, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras import backend as K


class LocalResponseNormalization(Layer):

    def __init__(self, n=5, alpha=0.0005, beta=0.75, k=2, **kwargs):

        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self._shape = None
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self._shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x, mask=None):
        averaged = 0

        if K.image_dim_ordering == "th":
             _, f, r, c = self._shape

        else:
            _, r, c, f = self._shape
            squared = K.square(x)
            pooled = K.pool2d(squared, (self.n, self.n), strides=(1, 1), padding="same", pool_mode="avg")

            if K.image_dim_ordering == "th":

                summed = K.sum(pooled, axis=1, keepdims=True)
                averaged = self.alpha * K.repeat_elements(summed, f, axis=1)

            else:

                summed = K.sum(pooled, axis=3, keepdims=True)
                averaged = self.alpha * K.repeat_elements(summed, f, axis=3)

        denom = K.pow(self.k + averaged, self.beta)
        return x / denom

    def get_output_shape_for(self, input_shape):
        return input_shape
