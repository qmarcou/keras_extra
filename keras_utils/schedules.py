from tensorflow import keras


def construct_increasing_lr_schedule(min_lr, step_increase):
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=min_lr,
        decay_steps=1,
        decay_rate=1.0 + step_increase,
        staircase=False)
    return lr_schedule
