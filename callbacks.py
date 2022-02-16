"""A collection of custom Keras callbacks."""
from tensorflow import keras


class BatchLossLRHistory(keras.callbacks.Callback):
    """A callback logging loss and learning rate for each batch."""

    # https://stackoverflow.com/questions/42392441/how-to-record-val-loss-and-loss-per-batch-in-keras
    def on_train_begin(self, logs={}):
        self.losses = []
        self.learning_rates = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.learning_rates.append(
            self.model.optimizer.lr(self.model.optimizer.iterations))
