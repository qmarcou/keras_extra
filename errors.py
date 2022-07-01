from tensorflow import errors


class KerasTunerSearchOOM(errors.ResourceExhaustedError):
    """A specific Exception to be thrown when Keras Tuner is out of memory
    due to Tensorflow's infamous memory leak """
    pass
