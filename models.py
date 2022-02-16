import keras_tuner
from tensorflow import keras
import keras_tuner as kt
from tensorflow.python.keras.callbacks import EarlyStopping


def sequential_multilabel_model(n_layers, layer_size, output_size,
                                input_size=None):
    # TODO return a compiled model
    if input_size is not None:
        input_layer = [
            keras.Input(shape=(input_size,), sparse=False, dtype=bool)]
    else:
        input_layer = []
    layers = input_layer + [keras.layers.Dense(units=layer_size,
                                               activation='relu',
                                               name="hiddenL" + str(i)) for i
                            in range(1, n_layers)]
    layers.append(
        keras.layers.Dense(units=output_size, name="outputL", activation=None))
    model = keras.Sequential(layers)
    # model.add_loss(keras.losses.CategoricalCrossentropy())
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-4),
             loss=keras.losses.BinaryCrossentropy(from_logits=True),
             metrics=["accuracy"])
    return model


class SequentialMultilabelHypermodel(kt.HyperModel):
    def __init__(self, output_size: int):
        self.output_size = output_size
        super(SequentialMultilabelHypermodel, self).__init__()

    def build(self, hp: keras_tuner.HyperParameters):
        return sequential_multilabel_model(hp.Int("num_layers", 1, 5),
                                           hp.Int("units", min_value=32,
                                                  max_value=2048,
                                                  step=128),
                                           output_size=self.output_size,
                                           input_size=None)

    def fit(self, hp: keras_tuner.HyperParameters,
            model: keras.models.Model, x, y,
            train_dev_frac, x_dev, y_dev, dev_metrics,
            earlystop_monitor: str = 'val_loss',
            **kwargs):
        early_stopping_callback = EarlyStopping(monitor=earlystop_monitor,
                                                min_delta=0.1,
                                                patience=5,
                                                verbose=1,
                                                restore_best_weights=True)
        callbacks = kwargs.get("callbacks", None)
        if callbacks is None:
            callbacks = []
        callbacks.append(early_stopping_callback)

        history = model.fit(x, y,
                            shuffle=True,
                            validation_split=train_dev_frac,
                            **kwargs)
        dev_data = (x_dev, y_dev)
        dev_y_pred = model.predict(x_dev)
        dev_metrics_dict = {}
        for metric in dev_metrics:
            dev_metrics_dict[metric.__name__] = metric(y_dev,dev_y_pred)
        return dev_metrics_dict
