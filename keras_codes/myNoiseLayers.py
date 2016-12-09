from __future__ import absolute_import
from keras.engine import Layer
from keras import backend as K
import numpy as np


class LaplacianNoise(Layer):
    '''Apply to the input an additive zero-centered Laplacian noise with
    diversity `b`.
    As it is a regularization layer, it is only active at training time.

    # Arguments
        b: float, diversity of the Laplacian noise

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __init__(self, b, **kwargs):
        self.supports_masking = True
        self.sigma = b
        self.uses_learning_phase = True
        super(LaplacianNoise, self).__init__(**kwargs)

    def call(self, x, mask=None):
        X = K.random_uniform(shape=K.shape(x),low=0,high=1)
        Y = K.random_uniform(shape=K.shape(x),low=0,high=1)
        noise = self.sigma * K.log(X/Y)
        noise_x = x + noise
        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(LaplacianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianNoise4tanh(Layer):
    '''Apply to the input an additive zero-centered Gaussian noise with
    standard deviation `sigma`. The corrupted data are forced to range
    between -1 and 1.

    As it is a regularization layer, it is only active at training time.

    # Arguments
        sigma: float, standard deviation of the noise distribution.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    '''
    def __init__(self, sigma, **kwargs):
        self.supports_masking = True
        self.sigma = sigma
        self.uses_learning_phase = True
        super(GaussianNoise4tanh, self).__init__(**kwargs)

    def call(self, x, mask=None):
        noise_x = x + K.random_normal(shape=K.shape(x),
                                      mean=0.,
                                      std=self.sigma)
        noise_x = K.minimum(K.maximum(noise_x,-1),1)
        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(GaussianNoise4tanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))