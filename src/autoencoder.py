from keras.layers import Input, Dense
from keras.models import Model

import math
import numpy as np


class Autoencoder:

    def __init__(self, input, num_node_list, rate=1e-4, decay=1e-6, batch=128):
        size_input = np.size(input)
        self.input = np.reshape(input.copy(), (size_input, 1))

        num_node_list_reverse = num_node_list.reverse()
        self.num_node_list = [size_input] + num_node_list + num_node_list_reverse[1:] + [size_input]
        self.autoencoder_layer_list = []

        self.rate = rate
        self.decay = decay
        self.batch = batch

    def build(self):
        pass

    def fit(self):
        pass

    def get_latent(self):
        pass

    def set_latent(self, latent=None):
        """Set the latent variables of the autoencoder.
        
        :param latent: [description], defaults to None
        :type latent: [type], optional
        """

        pass


if __name__ == "__main__":
    skeleton_input = np.random.normal(size=(5, 25))
    pass
    # skeleton_autoencoder = Autoencoder(input, [512, 256, 128, 64, 7])
