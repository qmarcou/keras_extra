from tensorflow import keras
import keras_tuner as kt
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy
import keras_utils.metrics
import copy


def sequential_multilabel_model(n_layers, layer_size, output_size,
                                input_size=None, batchnorm: bool = False,
                                dropout: float = 0, output_bias=None,
                                activation='relu',
                                loss=BinaryCrossentropy(from_logits=False),
                                learning_rate=2e-4):
    if input_size is not None:
        input_layer = [
            keras.Input(shape=(input_size,), sparse=False, dtype=bool)]
    else:
        input_layer = []
    hidden_layers = []
    # Use linear activation if batchnorm to scale the logits
    for i in range(1, n_layers):
        hidden_layers.append(keras.layers.Dense(units=layer_size,
                                                activation='linear',
                                                name="DenseHiddenL_" + str(i),
                                                use_bias=not batchnorm))
        if batchnorm:
            hidden_layers.append(keras.layers.BatchNormalization(
                name="batchnormL_" + str(i)))

        # Create activation after batchnorm to scale logits
        hidden_layers.append(keras.layers.Activation(activation=activation,
                                                     name=("activationL_" +
                                                           str(i))))

        if dropout != 0:
            hidden_layers.append(keras.layers.Dropout(dropout,
                                                      name="dropoutL_"
                                                           + str(i)))

    layers = input_layer + hidden_layers
    layers.append(
        keras.layers.Dense(units=output_size, name="outputL_",
                           activation='sigmoid'))
    model = keras.Sequential(layers)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=[keras_utils.metrics.Coverage(),
                           keras.metrics.BinaryAccuracy(),
                           keras.metrics.Precision(),
                           keras.metrics.AUC(),
                           keras.metrics.Recall()])
    return model


class SequentialMultilabelHypermodel(kt.HyperModel):
    def __init__(self, output_size: int, hp_kwargs: dict, **kwargs):
        self.output_size = output_size
        self.hp_kwargs = hp_kwargs
        self.build_kwargs = kwargs
        super(SequentialMultilabelHypermodel, self).__init__()

    def build(self, hp: kt.HyperParameters):
        hp_kwargs = copy.deepcopy(self.hp_kwargs)
        if hp_kwargs['use_dropout']['enable']:
            hp_kwargs['use_dropout'].pop('enable')
            if hp.Boolean(name="use_dropout", **hp_kwargs['use_dropout']):
                dropout_rate = hp.Float(name="dropout", **hp_kwargs['dropout'])
            else:
                dropout_rate = 0.0
        else:
            dropout_rate = 0.0
        if hp_kwargs['batchnorm']['enable']:
            hp_kwargs['batchnorm'].pop('enable')
            batchnorm = hp.Boolean(name="batchnorm", **hp_kwargs['batchnorm'])
        else:
            batchnorm = False
        return sequential_multilabel_model(hp.Int("num_layers",
                                                  **hp_kwargs['num_layers']),
                                           hp.Int("units",
                                                  **hp_kwargs['units']),
                                           output_size=self.output_size,
                                           input_size=None,
                                           batchnorm=batchnorm,
                                           dropout=dropout_rate,
                                           activation=hp.Choice('activation',
                                                                **hp_kwargs[
                                                                    'activation']),
                                           **self.build_kwargs)

    def fit(self, hp: kt.HyperParameters,
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
