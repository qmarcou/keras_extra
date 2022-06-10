"""A collection of custom Keras callbacks."""
from tensorflow import keras
import tensorflow as tf


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


class DataEvaluator(keras.callbacks.Callback):
    """A callback to evaluate a model on specific data."""

    def __init__(self, x, y,
                 eval_prefix="dataEval_", **kwargs):
        self.x_data = x
        self.y_data = y
        self.eval_prefix = eval_prefix
        self.eval_kwargs = kwargs
        if kwargs.get("return_dict") is not None:
            raise AttributeError("return_dict cannot be specified when used "
                                 "in the DataEvaluator callback")

    def on_epoch_end(self, epoch, logs=None):
        # Evaluate the model on the provided data
        eval_output = self.model.evaluate(
            x=self.x_data,
            y=self.y_data,
            return_dict=True,
            **self.eval_kwargs
        )
        # Add the prefix to all keys of the dict
        eval_output = {self.eval_prefix + str(key): val for key, val in
                       eval_output.items()}

        # Check that none of the new keys overshadow the previous ones
        if len(set(eval_output.keys())
                       .intersection(logs.keys())) > 0:
            raise RuntimeError("Names resulting from the DataEvaluator "
                               "overshadows names from the logs dict, "
                               "pick a different eval_prefix")
        # Update epoch_logs dict
        logs.update(eval_output)
