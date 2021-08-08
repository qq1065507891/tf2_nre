import tensorflow as tf

from src.models.TextCNN import TextCNN


def init_model(config, model_name, vocab_size):
    model = TextCNN(config, model_name, vocab_size).build_model()
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        metrics=['accuracy']
    )
    return model


def load_model(config, model_name, vocab_size):
    model = init_model(config, model_name, vocab_size)
    model.load_weights(config.model_pkl)
    return model


def predict():
    pass