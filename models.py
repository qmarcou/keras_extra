from tensorflow import keras


def create_sequential_model(n_layers, layer_size, output_size,
                            input_size=None):
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
    return model
