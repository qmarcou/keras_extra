"""A collection of custom Keras callbacks."""
from tensorflow import keras
import tensorflow as tf
from libs import dtypecheck


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
        self.eval_kwargs = {'verbose': 0}  # set default verbosity to 0
        self.eval_kwargs.update(kwargs)  # verbosity can be overriden by kwargs
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


class CompoundMetric(keras.callbacks.Callback):
    """A callback enabling to compute a metric by averaging existing ones."""
    def __init__(self, metrics_dict: dict,
                 met_prefixes: str | list[str] = None,
                 compound_metric_name: str = None):
        # TODO check that its a {str:float} dict
        build_metric_name = False
        self._metrics_dict = metrics_dict
        self.comp_met_name = compound_metric_name

        if met_prefixes is None:
            self.met_prefixes = [""]
        else:
            if not dtypecheck.is_str_or_strlist(met_prefixes):
                raise TypeError("met_prefixes must be a str or list of str")
            else:
                if isinstance(met_prefixes, str):
                    met_prefixes = [met_prefixes]
                self.met_prefixes = met_prefixes
        if compound_metric_name is None:
            self.comp_met_name = ''
            build_metric_name = True

        for key in metrics_dict:
            if build_metric_name:
                self.comp_met_name += key + str(metrics_dict[key]) + "_"

        if build_metric_name:
            # remove the trailing underscore
            self.comp_met_name = self.comp_met_name[0:-1]

    def on_epoch_end(self, epoch, logs=None):
        for prefix in self.met_prefixes:
            aggregator = 0.0
            normalizer = 0.0
            for key in self._metrics_dict:
                aggregator += self._metrics_dict[key] * logs[prefix + key]
                normalizer += self._metrics_dict[key]
            aggregator /= normalizer

            # Update epoch_logs dict
            logs.update({prefix + self.comp_met_name: aggregator})
