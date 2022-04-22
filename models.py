from tensorflow import keras
import keras_tuner as kt
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy
import keras_utils.metrics
import copy


# Some useful ressources:
# Multi-output: https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras


class SequentialPreOutputLoss(keras.Sequential):
    """
    A class building a Sequential model with separate loss and output.

    A wrapper class allowing to use the simple Keras Sequential model interface
    for model building but adding the flexibility of using a layer different
    from the last one to compute the prediction loss (specified in
    model.compile).
    The only difference with a Sequential model is in the ability to have 2
    outputs: one for actual outputs (and metrics computation), another for
    prediction loss computation. If the 'loss_layer_name' argument is not
    provided the model will behave like a normal
    """

    def __init__(self, layers=None, name=None,
                 loss_layer_name: str = None):
        # Instantiate a Sequential model from parameters
        self.loss_layer_name = loss_layer_name
        self._loss_layer_added = False
        self._loss_tensor_output = None
        self._loss_layer: keras.layers.Layer = None
        self.output_layer_name = None
        super(SequentialPreOutputLoss, self).__init__(layers=layers,
                                                      name=name)

    def _add_loss_output(self):
        # To get the Sequential API working the final output layer tensor must
        # remain outputs[0], the loss one will always be outputs[1]
        # Ensure the first output exists, and add loss as second one
        assert len(self.outputs) == 1
        if self.loss_layer_name is not None and \
                self.loss_layer_name != self.output_layer_name:
            self.outputs.append(self._loss_tensor_output)

    def add(self, layer):
        super(SequentialPreOutputLoss, self).add(layer)
        if self.loss_layer_name is not None and \
                layer.name == self.loss_layer_name:
            self._loss_layer_added = True
            self._loss_layer = layer
            if self.outputs is not None:
                # If no input shape has been given for the network output
                # tensors are initialized at build time
                self._loss_tensor_output = self.outputs[0]

        if self.outputs is not None:
            self.output_layer_name = layer.name
            # Add back the 2nd output
            if self._loss_layer_added:
                self._add_loss_output()

        # Require to rebuild the model to trigger correct filling of
        # output_names and output_layers attributes from the functional API
        # If not filled correctly the API will fail to map the loss_layer_name
        # with the actual layer
        self.built = False

    def build(self, input_shape=None):
        super(SequentialPreOutputLoss, self).build(input_shape=input_shape)
        # In case no Input or batch_shape had been provided output tensors
        # had not been created
        if self.loss_layer_name is not None and \
                self._loss_tensor_output is None:
            self.output_layer_name = self.layers[-1].name
            self._loss_tensor_output = self._loss_layer.output
            self._add_loss_output()

    def compile(self,
                loss=None,
                metrics=None,
                *args,
                **kwargs):
        if not self.built:
            if self.input_shape is not None:
                self.build()
            else:
                raise RuntimeError("Cannot build the model with unknown input "
                                   "shape. Compiling SequentialPreOutputLoss "
                                   "before the model is built will result in "
                                   "errors at runtime. Build the model before "
                                   "compiling.")

        # Check that the specified loss layer has been added to the model
        if self.loss_layer_name is not None and not self._loss_layer_added:
            raise ValueError("The layer with specified loss_layer_name has "
                             "not been added to the model")

        # transparently assign loss and metrics computation to the correct
        # layers in case loss and metrics have been passed as lists
        if not isinstance(loss, dict) and self.loss_layer_name is not None:
            loss = {self.loss_layer_name: loss}

        if not isinstance(metrics, dict) and self.loss_layer_name is not None:
            metrics = {self.output_layer_name: metrics}
        super(keras.Sequential, self).compile(*args,
                                              loss=loss, metrics=metrics,
                                              **kwargs)

    def get_config(self):
        config = {'loss_layer_name': self.loss_layer_name}
        base_config = super(SequentialPreOutputLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def sequential_multilabel_model(n_layers, layer_size, output_size, input_size,
                                detached_loss=False,
                                batchnorm: bool = False,
                                dropout: float = 0, output_bias=None,
                                activation='relu', output_activation='sigmoid',
                                output_layer: keras.layers.Layer = None,
                                loss=BinaryCrossentropy,
                                loss_kwargs={'from_logits': False},
                                metrics=(keras_utils.metrics.Coverage(),
                                         keras.metrics.BinaryAccuracy(),
                                         keras.metrics.Precision(),
                                         keras.metrics.AUC(),
                                         keras.metrics.Recall()),
                                learning_rate=2e-4,
                                compile_kwargs: dict = {}):
    """
    A function to build a sequential model from simple parameters.

    Parameters
    ----------
    n_layers
    layer_size
    output_size
    detached_loss
    input_size
    batchnorm
    dropout
    output_bias
    activation
    output_activation
    output_layer: if detached loss, the provided layer will sit just below the
        activation layer implied by output_activation
    loss
    loss_kwargs
    metrics
    learning_rate
    compile_kwargs

    Returns
    -------

    """
    if input_size is not None:
        input_layer = [
            keras.layers.InputLayer(input_shape=(input_size,),
                                    sparse=False, dtype=bool)]
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
    if detached_loss:
        if output_layer is not None:
            # Append the provided layer as the last layer
            layers.append(output_layer)
            loss_layer_name = output_layer.name
        else:
            # Add a final linear Dense layer
            loss_layer_name = "denseL_" + str(n_layers)
            layers.append(
                keras.layers.Dense(units=output_size, name=loss_layer_name,
                                   activation='linear'))

        if isinstance(output_activation, keras.layers.Activation):
            layers.append(output_activation)
        elif isinstance(output_activation, str):
            layers.append(keras.layers.Activation(activation=output_activation,
                                                  name="OutputL"))
        else:
            raise ValueError("output_activation must be either a string or"
                             "an instance of keras.Activation")

        model = SequentialPreOutputLoss(layers=layers,
                                        loss_layer_name=loss_layer_name)
    else:
        if output_layer is not None:
            # Append the provided layer as the last layer
            layers.append(output_layer)
        else:
            # Add a dense layer with sigmoid activation
            layers.append(
                keras.layers.Dense(units=output_size, name="outputL_",
                                   activation='sigmoid'))
        model = keras.Sequential(layers)

    model.build()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=loss(**loss_kwargs),
                  metrics=metrics,
                  **compile_kwargs)
    return model


class SequentialMultilabelHypermodel(kt.HyperModel):
    def __init__(self, output_size: int, hp_kwargs: dict, **kwargs):
        self.output_size = output_size
        self.hp_kwargs = hp_kwargs
        self.build_kwargs = kwargs
        super(SequentialMultilabelHypermodel, self).__init__()

    def build(self, hp: kt.HyperParameters):
        hp_kwargs = copy.deepcopy(self.hp_kwargs)
        build_kwargs = copy.deepcopy(self.build_kwargs)
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

        output_layer = None
        if hp_kwargs.get('outputHierL2Reg'):
            if hp_kwargs['outputHierL2Reg']['enable']:
                hp_kwargs['outputHierL2Reg'].pop('enable')
                adj_mat = hp_kwargs['outputHierL2Reg'].pop('adjacency_matrix')
                output_layer = keras_utils.layers.DenseHierL2Reg(
                    units=self.output_size,
                    adjacency_matrix=adj_mat,
                    hier_side='out',
                    regularization_factor=hp.Float(name="hierL2reg_factor",
                                                   **hp_kwargs['outputHierL2Reg']),
                    tree_like=True
                )

        loss_class = self.build_kwargs.get('loss')
        # Custom hyperparameters for peculiar metrics
        if loss_class is not None and loss_class.__name__ == \
                'WeightedBinaryCrossentropy':
            if hp_kwargs['class_weight_cap']['enable']:
                hp_kwargs['class_weight_cap'].pop('enable')
                # Create a hyperparameter for class weight cap
                build_kwargs['loss_kwargs']['class_weights'] = (
                    build_kwargs['loss_kwargs']['class_weights']
                        .clip(max=hp.Float(**hp_kwargs['class_weight_cap'])))

        return sequential_multilabel_model(hp.Int("num_layers",
                                                  **hp_kwargs['num_layers']),
                                           hp.Int("units",
                                                  **hp_kwargs['units']),
                                           output_size=self.output_size,
                                           batchnorm=batchnorm,
                                           dropout=dropout_rate,
                                           activation=hp.Choice('activation',
                                                                **hp_kwargs[
                                                                    'activation']),
                                           output_layer=output_layer
                                           **build_kwargs)

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
