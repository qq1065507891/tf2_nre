import os
import tensorflow as tf
import time

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.utils.process import process
from src.utils.utils import load_pkl, training_curve, get_time_idf, load_embedding, get_embedding_matrix
from src.utils.dataset import Dateset
from src.models.TextCNN import TextCNN


def init_model(config, model_name, vocab_size, embedding_matrix=None):
    model = TextCNN(config, model_name, vocab_size, embedding_matrix).build_model()
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        metrics=['accuracy']
    )
    return model


def load_model(config, model_name, vocab_size, embedding_matrix=None):
    model = init_model(config, model_name, vocab_size, embedding_matrix=embedding_matrix)
    model.load_weights(config.model_pkl)
    return model


def train(config, model_name):
    if not os.path.exists(config.out_path):
        process(config)

    train_data_path = os.path.join(config.out_path, 'train.pkl')
    test_data_path = os.path.join(config.out_path, 'test.pkl')
    vocab_path = os.path.join(config.out_path, 'vocab.pkl')
    embedding_path = os.path.join(config.data_path, 'vec.txt')

    embedding = load_embedding(embedding_path)
    train_data = load_pkl(train_data_path, 'train_data')
    test_data = load_pkl(test_data_path, 'test_data')
    vocab = load_pkl(vocab_path, 'vocab')

    embedding_matrix = get_embedding_matrix(config, embedding, vocab)
    vocab_size = len(vocab.word2id)
    start_time = time.time()

    dataset = Dateset(config)
    train_iter = dataset.iter_dataset(train_data).shuffle(buffer_size=50).\
        batch(config.batch_size, drop_remainder=True).repeat(config.batch_size)

    test_iter = dataset.iter_dataset(test_data).batch(config.batch_size, drop_remainder=True).repeat(config.batch_size)

    print(len(list(train_iter)) // config.batch_size)
    end_time = get_time_idf(start_time)

    print(f'train_data: {len(list(train_iter))}, test_data: {len(list(test_iter))}, 用时: {end_time}')

    print('加载模型')
    if os.path.exists(config.model_pkl):
        model = load_model(config, model_name, vocab_size, embedding_matrix)
    else:
        model = init_model(config, model_name, vocab_size, embedding_matrix)

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='max'),
        ModelCheckpoint(config.model_pkl, monitor='val_accuracy', verbose=1,
                        save_weights_only=True, save_best_only=True, mode='max'),
    ]
    print('开始训练')
    start_time = time.time()
    history = model.fit(
        train_iter,
        epochs=config.epochs,
        steps_per_epoch=32,
        validation_data=test_iter,
        validation_steps=config.batch_size,
        callbacks=callbacks
    )
    end_time = get_time_idf(start_time)
    print(f'训练完成, 用时: {end_time}')
    training_curve(history.history['loss'], history.history['accuracy'],
                   history.history['val_loss'], history.history['val_accuracy'])
