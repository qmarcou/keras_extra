from tensorflow import errors


class KerasTunerSearchOOM(errors.ResourceExhaustedError):
    """A specific Exception to be thrown when Keras Tuner is out of memory
    due to Tensorflow's infamous memory leak """
    pass


class KerasTunerSearchOOMSameTrial(errors.ResourceExhaustedError):
    """Another specific Exception to be thrown when Keras Tuner is out of memory
    due to Tensorflow's infamous memory leak, and it happens on a previously
    erroneous trial."""
    pass
