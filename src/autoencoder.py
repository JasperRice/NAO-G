from keras.layers import Input, Dense
from keras.models import Model

class Autoencoder:

    def __init__(self, input, num_node, rate=1e-4, decay=1e-6, batch=128):
        self.num_layer = len(num_node)
        self.num_node = num_node
        self.input_layer = Input()
        self.encoder_layers = []
        self.decoder_layers = []

    def build(self):
        pass

    def fit(self):
        pass

    def get_latent(self):
        pass

    def set_latent(self):
        pass


if __name__ == "__main__":
    pass
    # skeleton_autoencoder = Autoencoder(input, [512, 256, 128, 64, 7])
