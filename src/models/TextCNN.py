import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Input


class TextCNN(object):
    def __init__(self, config, model_name, vocab_size, embedding_matrix=None):
        self.config = config
        self.model_name = model_name
        if embedding_matrix:
            self.word_embed = Embedding(vocab_size, config.word_dim, weights=[embedding_matrix], trainable=False)
        else:
            self.word_embed = Embedding(vocab_size, config.word_dim)
        self.head_embed = Embedding(config.pos_size, config.word_dim)
        self.tail_embed = Embedding(config.pos_size, config.word_dim)

        self.conv1 = Conv1D(filters=config.out_channels, kernel_size=2, padding='same', activation='relu', name='cnn1')
        self.conv2 = Conv1D(filters=config.out_channels, kernel_size=3, padding='same', activation='relu', name='cnn2')
        self.conv3 = Conv1D(filters=config.out_channels, kernel_size=4, padding='same', activation='relu', name='cnn3')
        self.flatten = Flatten()

        self.maxpooling = MaxPooling1D(pool_size=2, name='maxpooling')
        self.fc1 = Dense(config.fc1_channels, activation='relu', name='fc1')
        self.out = Dense(config.relation_type, activation='softmax', name='output')

        self.dropout = Dropout(config.dropout)

    def build_model(self):
        word_input = Input(shape=(self.config.max_len,), name='word_input')
        head_input = Input(shape=(self.config.max_len,), name='head_input')
        tail_input = Input(shape=(self.config.max_len,), name='tail_input')

        word_embed = self.word_embed(word_input)
        head_embed = self.head_embed(head_input)
        tail_embed = self.tail_embed(tail_input)

        input_embed = tf.keras.layers.concatenate([word_embed, head_embed, tail_embed], axis=-1)

        cnn1 = self.conv1(input_embed)
        cnn1 = self.maxpooling(cnn1)

        cnn2 = self.conv2(input_embed)
        cnn2 = self.maxpooling(cnn2)

        cnn3 = self.conv3(input_embed)
        cnn3 = self.maxpooling(cnn3)

        cnn = tf.keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)

        flatten = self.flatten(cnn)
        flatten = self.dropout(flatten)
        fc1 = self.fc1(flatten)
        fc1 = self.dropout(fc1)
        output = self.out(fc1)

        model = tf.keras.Model(inputs=[word_input, head_input, tail_input], outputs=output, name=self.model_name)
        model.summary()
        return model

