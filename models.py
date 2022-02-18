import keras_tuner
from tensorflow import keras
import keras_tuner as kt
from tensorflow.python.keras.callbacks import EarlyStopping

import keras_utils.metrics


def sequential_multilabel_model(n_layers, layer_size, output_size,
                                input_size=None, batchnorm: bool = False,
                                dropout: float = 0, output_bias=None,
                                activation='relu'):
    if input_size is not None:
        input_layer = [
            keras.Input(shape=(input_size,), sparse=False, dtype=bool)]
    else:
        input_layer = []
    hidden_layers = []
    for i in range(1, n_layers):
        hidden_layers.append(keras.layers.Dense(units=layer_size,
                                                activation=activation,
                                                name="hiddenL_" + str(i),
                                                use_bias=not batchnorm))
        if batchnorm:
            hidden_layers.append(keras.layers.BatchNormalization(
                name="batchnormL_" + str(i)))
        if dropout != 0:
            hidden_layers.append(keras.layers.Dropout(dropout,
                                                      name="dropoutL_"
                                                           + str(i)))

    layers = input_layer + hidden_layers
    layers.append(
        keras.layers.Dense(units=output_size, name="outputL_",
                           activation='sigmoid'))
    model = keras.Sequential(layers)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[keras_utils.metrics.Coverage(),
                           keras.metrics.BinaryAccuracy(),
                           keras.metrics.Precision(),
                           keras.metrics.AUC(),
                           keras.metrics.Recall()])
    return model


class SequentialMultilabelHypermodel(kt.HyperModel):
    def __init__(self, output_size: int):
        self.output_size = output_size
        super(SequentialMultilabelHypermodel, self).__init__()

    def build(self, hp: keras_tuner.HyperParameters):

        if hp.Boolean(name="use_dropout", default=True):
            dropout_rate = hp.Float(name="dropout", min_value=1e-3,
                                    max_value=.4,
                                    parent_name="use_dropout",
                                    parent_values=True,
                                    sampling='log', default=0)
        else:
            dropout_rate = 0.0
        return sequential_multilabel_model(hp.Int("num_layers", 1, 10),
                                           hp.Int("units", min_value=32,
                                                  max_value=2048,
                                                  step=128),
                                           output_size=self.output_size,
                                           input_size=None,
                                           batchnorm=hp.Boolean("batchnorm"),
                                           dropout=dropout_rate,
                                           activation=hp.Choice('activation',
                                                                values=['relu',
                                                                        'tanh']))

    def fit(self, hp: keras_tuner.HyperParameters,
            model: keras.models.Model, x, y,
            train_dev_frac, x_dev, y_dev, dev_metrics,
            earlystop_monitor: str = 'val_loss',
            **kwargs):
        early_stopping_callback = EarlyStopping(monitor=earlystop_monitor,
                                                min_delta=0.1,
                                                patience=10,
                                                verbose=1,
                                                restore_best_weights=True)
        callbacks = kwargs.get("callbacks", None)
        if callbacks is None:
            callbacks = []
        callbacks.append(early_stopping_callback)
        kwargs["callbacks"] = callbacks
        history = model.fit(x, y,
                            shuffle=True,
                            validation_split=train_dev_frac,
                            **kwargs)
        return history
        dev_data = (x_dev, y_dev)
        # dev_y_pred = model.predict(x_dev)
        # dev_metrics_dict = {}
        # for metric in dev_metrics:
        #     dev_metrics_dict['dev_'+metric.__name__] = metric(y_dev,
        #                                                       dev_y_pred)
        # return dev_metrics_dict
