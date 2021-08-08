import os
import codecs
import pickle
import time
import matplotlib.pyplot as plt

from datetime import timedelta


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_file(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def save_pkl(path, obj, obj_name):
    print(f"{obj_name} save in {path}")
    with codecs.open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pkl(path, obj_name):
    print(f'load {obj_name} in {path}')
    with codecs.open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def training_curve(loss, acc, val_loss=None, val_acc=None):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(loss, color='r', label='Training Loss')
    if val_loss is not None:
        ax[0].plot(val_loss, color='g', label='Validation Loss')
    ax[0].legend(loc='best', shadow=True)
    ax[0].grid(True)

    ax[1].plot(acc, color='r', label='Training Accuracy')
    if val_loss is not None:
        ax[1].plot(val_acc, color='g', label='Validation Accuracy')
    ax[1].legend(loc='best', shadow=True)
    ax[1].grid(True)
    plt.show()


def get_time_idf(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return: 返回使用多长时间
    """
    end_time = time.time()
    time_idf = end_time - start_time
    return timedelta(seconds=int(round(time_idf)))
