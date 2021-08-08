import os
import jieba
import time
import codecs

from tensorflow.keras.utils import to_categorical

from src.utils.utils import read_file, ensure_dir, save_pkl, load_pkl, get_time_idf
from src.utils.config import config
from src.utils.Enconde import TextEnconde


def build_vocab(texts, out_path):
    vocab = TextEnconde()
    for text in texts:
        vocab.add_sentences(text)
    vocab.trim(config.min_freq)

    ensure_dir(out_path)

    vocab_path = os.path.join(out_path, 'vocab.pkl')
    vocab_txt = os.path.join(out_path, 'vocab.txt')

    save_pkl(vocab_path, vocab, 'vocab')

    with codecs.open(vocab_txt, 'w', encoding='utf-8') as f:
        f.write(os.linesep.join([word for word in vocab.word2id.keys()]))

    return vocab, vocab_txt


def get_data(datas):
    entity_ones, entity_twos = [], []
    texts, relations = [], []
    pos = []
    jieba.add_word('HEAD')
    jieba.add_word('TAIL')
    for data in datas:
        all_datas = data.strip().split('\t')
        entity_one = all_datas[0]
        entity_two = all_datas[1]
        relation = all_datas[2]
        text = all_datas[3]
        if len(entity_two) < len(entity_one):
            text = text.replace(entity_one, 'HEAD').replace(entity_two, 'TAIL')
        else:
            text = text.replace(entity_two, 'TAIL').replace(entity_one, 'HEAD')
        text = jieba.lcut(text)

        head_pos, tail_pos = text.index('HEAD'), text.index('TAIL')

        text[head_pos] = entity_one
        text[tail_pos] = entity_two

        entity_ones.append(entity_one)
        entity_twos.append(entity_two)
        relations.append(relation)
        texts.append(text)
        pos.append([head_pos, tail_pos])

    return entity_ones, entity_twos, relations, pos, texts


def get_pos_feature(sent_len, entities_pos, entity_len, pos_limit):
    """
    获取位置编码
    :param sent_len:
    :param entities_pos:
    :param entity_len:
    :param pos_limit:
    :return:
    """
    left = list(range(-entities_pos, 0))
    middle = [0] * entity_len
    right = list(range(1, sent_len - entities_pos - entity_len + 1))
    pos = left + middle + right

    for i, p in enumerate(pos):
        if p > pos_limit:
            pos[i] = pos_limit
        if p < -pos_limit:
            pos[i] = -pos_limit
    pos = [p + pos_limit + 1 for p in pos]

    return pos


def get_mask_feature(sent_len, entities_pos):
    """
    获取mask编码
    :param sent_len:
    :param entities_pos:
    :return:
    """
    left = [1] * (entities_pos[0] + 1)
    middle = [2] * (entities_pos[1] - entities_pos[0] - 1)
    right = [3] * (sent_len - entities_pos[1])
    return left + middle + right


def build_data(datas, vocab):
    sents = []
    head_pos = []
    tail_pos = []
    mask_pos = []

    entity_ones, entity_twos, pos, texts = datas
    for i, sent in enumerate(texts):
        text = [vocab.word2id.get(word, 0) for word in sent]
        head, tail = int(pos[i][0]), int(pos[i][1])
        entities_pos = [head, tail] if tail > head else [tail, head]
        head_p = get_pos_feature(len(sent), head, 1, config.pos_limit)
        tail_p = get_pos_feature(len(sent), tail, 1, config.pos_limit)
        mask_p = get_mask_feature(len(sent), entities_pos)

        sents.append(text)
        head_pos.append(head_p)
        tail_pos.append(tail_p)
        mask_pos.append(mask_p)
    return sents, head_pos, tail_pos, mask_pos


def get_relation2id(config):
    path = os.path.join(config.data_path, 'relation2id.txt')
    pkl_path = os.path.join(config.out_path, 'relation2id.pkl')
    if os.path.exists(pkl_path):
        relation2id = load_pkl(pkl_path, 'relation2id')
    else:
        relation2id = {}
        with codecs.open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                lines = line.strip().split(' ')
                relation2id[lines[0]] = int(lines[1])
        save_pkl(pkl_path, relation2id, 'relation2id')
    return relation2id


def relation_tokenize(config, relation):
    relation2id = get_relation2id(config)
    relations = []
    for rel in relation:
        relations.append(relation2id.get(rel, 0))
    relations = to_categorical(relations, num_classes=config.relation_type, dtype='int32')
    return relations


def process(config):
    ensure_dir(config.out_path)

    print('数据预处理开始')
    start_time = time.time()

    print('读取数据')
    train_data = read_file(os.path.join(config.data_path, 'train.txt'))
    test_data = read_file(os.path.join(config.data_path, 'test.txt'))

    print('预处理数据')
    train_entity_ones, train_entity_twos, train_relations, train_pos, train_texts = get_data(train_data)
    test_entity_ones, test_entity_twos, test_relations, test_pos, test_texts = get_data(test_data)
    train_data = [train_entity_ones, train_entity_twos, train_pos, train_texts]
    test_data = [test_entity_ones, test_entity_twos, test_pos, test_texts]

    all_texts = test_texts + train_texts

    print('建立词表')
    vocab, vocab_path = build_vocab(all_texts, config.out_path)

    print('构建模型数据')
    train_sents, train_head_pos, train_tail_pos, train_mask = build_data(train_data, vocab)
    test_sents, test_head_pos, test_tail_pos, test_mask = build_data(test_data, vocab)

    print('构建关系型数据')
    train_relations_token = relation_tokenize(config, train_relations)
    test_relations_token = relation_tokenize(config, test_relations)

    train_data = (train_sents, train_head_pos, train_tail_pos, train_mask, train_relations_token)

    test_data = (test_sents, test_head_pos, test_tail_pos, test_mask, test_relations_token)

    train_data_path = os.path.join(config.out_path, 'train.pkl')
    test_data_path = os.path.join(config.out_path, 'test.pkl')

    save_pkl(train_data_path, train_data, 'train_data')
    save_pkl(test_data_path, test_data, 'test_data')

    end_time = get_time_idf(start_time)
    print(f'数据预处理结束, 用时{end_time}')


if __name__ == '__main__':
    process(config)






