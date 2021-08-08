import tensorflow as tf


class Dateset(object):
    def __init__(self, config):
        self.config = config

    def pad_suqentens(self, datas):
        pad = []
        for data in datas:
            if len(data) > self.config.max_len:
                pad.append(data[:self.config.max_len])
            else:
                data = data + [0] * (self.config.max_len - len(data))
                pad.append(data)
        return pad

    def name_to_dict(self, sents, head_pos, tail_pos, relations_token):
        return {
            'word_input': sents,
            'head_input': head_pos,
            'tail_input': tail_pos,
        }, relations_token

    def iter_dataset(self, datas):
        sents, head_pos, tail_pos, mask, relations_token = datas
        sents = self.pad_suqentens(sents)
        head_pos = self.pad_suqentens(head_pos)
        tail_pos = self.pad_suqentens(tail_pos)
        data_iter = tf.data.Dataset.from_tensor_slices((sents, head_pos,
                                                        tail_pos, relations_token)).map(self.name_to_dict)
        return data_iter
