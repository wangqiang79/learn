# -*- coding:utf-8 -*-
# Created by LuoJie at 11/22/19
from gensim.models.word2vec import LineSentence, Word2Vec
import numpy as np
import codecs
# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_word2vec_file(save_wv_model_path):
    # 保存词向量模型
    wv_model = Word2Vec.load(save_wv_model_path)
    embedding_matrix = wv_model.wv.vectors
    return embedding_matrix


def get_vocab(save_wv_model_path):
    # 保存词向量模型
    wv_model = Word2Vec.load(save_wv_model_path)
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    return vocab, reverse_vocab


def get_embedding_matrix(w2v_model):
    vocab_size = len(w2v_model.wv.vocab)
    embedding_dim = len(w2v_model.wv['<START>'])
    print('vocab_size, embedding_dim:', vocab_size, embedding_dim)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i in range(vocab_size):
        embedding_matrix[i, :] = w2v_model.wv[w2v_model.wv.index2word[i]]
        embedding_matrix = embedding_matrix.astype('float32')
    assert embedding_matrix.shape == (vocab_size, embedding_dim)
    np.savetxt('embedding_matrix.txt', embedding_matrix, fmt='%0.8f')
    print('embedding matrix extracted')
    return embedding_matrix


def build_vocab(vocab):
    """
    :param vocab:词表
    :return: 处理后的词表
    """
    start_token = u"<s>"
    end_token = u"<e>"
    unk_token = u"<unk>"

    # 按索引排序
    vocab = sorted([(vocab[i].index, i) for i in vocab])
    # 排序后的词
    sorted_words = [word for index, word in vocab]
    # 拼接标志位的词
    sorted_words = [start_token, end_token, unk_token] + sorted_words

    # 构建索引表
    vocab = {index: word for index, word in enumerate(sorted_words)}
    reverse_vocab = {word: index for index, word in enumerate(sorted_words)}
    return vocab, reverse_vocab
