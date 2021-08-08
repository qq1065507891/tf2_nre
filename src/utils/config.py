import os

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]


class Config(object):
    data_path = os.path.join(root_path, 'data')
    out_path = os.path.join(root_path, 'data/out')

    model_name = 'TextCNN'
    model_pkl = os.path.join(root_path, 'model/text_cnn.h5')

    embedding = True

    epochs = 100
    batch_size = 8
    learning_rate = 0.001

    min_freq = 2
    relation_type = 12
    pos_limit = 50

    pos_size = 102  # 2*pos_limit + 2
    word_dim = 100

    dropout = 0.5
    out_channels = 256
    fc1_channels = 512
    max_len = 64


config = Config()
