# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import re
import jieba
import pandas as pd
from utils.multi_proc_utils import parallelize
from utils.config import train_seg_path, test_seg_path, merger_seg_path, user_dict, train_x_seg_path, test_x_seg_path, \
    train_x_pad_path, train_y_pad_path, test_x_pad_path, wv_train_epochs
from utils.config import stop_word_path, train_data_path, test_data_path
import codecs

from gensim.models.word2vec import LineSentence, Word2Vec
import numpy as np

# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from utils.config import save_wv_model_path

# 自定义词表
jieba.load_userdict(user_dict)


def build_dataset(train_data_path, test_data_path):
    '''
    数据加载+预处理
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return: 训练数据 测试数据  合并后的数据
    '''
    # 1.加载数据
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    print('train data size {},test data size {}'.format(len(train_df), len(test_df)))

    # 2. 空值填充
    train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)

    # 3.多进程, 批量数据处理
    train_df = parallelize(train_df, sentences_proc)
    test_df = parallelize(test_df, sentences_proc)

    # 4. 合并训练测试集合
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df), len(test_df),
                                                                               len(merged_df)))

    # 5.保存处理好的 训练 测试集合
    train_df = train_df.drop(['merged'], axis=1)
    test_df = test_df.drop(['merged'], axis=1)
    train_df.to_csv(train_seg_path, index=None, header=True)
    test_df.to_csv(test_seg_path, index=None, header=True)

    # 6. 保存合并数据
    merged_df.to_csv(merger_seg_path, index=None, header=False)

    # 7. 训练词向量
    print('start build w2v model')
    wv_model = Word2Vec(LineSentence(merger_seg_path), size=300, negative=5, workers=8, iter=wv_train_epochs, window=3,
                        min_count=5)

    # 8. 分离数据和标签
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 9. 填充开始结束符号,未知词填充 oov, 长度填充
    # 使用GenSim训练得出的vocab
    vocab = wv_model.wv.vocab

    # 训练集X处理
    # 获取适当的最大长度
    train_x_max_len = get_max_len(train_df['X'])
    test_X_max_len = get_max_len(test_df['X'])
    X_max_len = max(train_x_max_len, test_X_max_len)
    train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))

    # 测试集X处理
    # 获取适当的最大长度
    test_df['X'] = test_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))

    # 训练集Y处理
    # 获取适当的最大长度
    train_y_max_len = get_max_len(train_df['Report'])
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_proc(x, train_y_max_len, vocab))

    # 10. 保存pad oov处理后的,数据和标签
    train_df['X'].to_csv(train_x_pad_path, index=None, header=False)
    train_df['Y'].to_csv(train_y_pad_path, index=None, header=False)
    test_df['X'].to_csv(test_x_pad_path, index=None, header=False)

    # 11. 词向量再次训练
    print('start retrain w2v model')
    wv_model.build_vocab(LineSentence(train_x_pad_path), update=True)
    wv_model.train(LineSentence(train_x_pad_path), epochs=wv_train_epochs, total_examples=wv_model.corpus_count)
    print('1/3')
    wv_model.build_vocab(LineSentence(train_y_pad_path), update=True)
    wv_model.train(LineSentence(train_y_pad_path), epochs=wv_train_epochs, total_examples=wv_model.corpus_count)
    print('2/3')
    wv_model.build_vocab(LineSentence(test_x_pad_path), update=True)
    wv_model.train(LineSentence(test_x_pad_path), epochs=wv_train_epochs, total_examples=wv_model.corpus_count)

    # 保存词向量模型
    wv_model.save(save_wv_model_path)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))

    return train_df['X'], train_df['Y'], test_df['X'], wv_model


def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    max_lens = data.apply(lambda x: x.count(' '))
    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def pad_proc(sentence, max_len, vocab):
    '''
    < start > < end > < pad > < unk > max_lens
    '''
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 3. 填充< start > < end >
    sentence = ['<START>'] + sentence + ['<STOP>']
    # 4. 判断长度，填充　< pad >
    sentence = sentence + ['<PAD>'] * (max_len + 2 - len(words))
    return ' '.join(sentence)


def load_word2vec_file(word2vec_file):
    '''
    加载词向量
    :param word2vec_file:词向量路径
    :return: 词向量字典, 维度
    '''
    word2vec_dict = {}
    input = codecs.open(word2vec_file, "r", "utf-8")
    lines = input.readlines()
    input.close()
    word_num, dim = lines[0].split(" ")
    dim = int(dim)

    lines = lines[1:]
    for l in lines:
        l = l.strip()
        tokens = l.split(" ")
        if len(tokens) != dim + 1:
            continue
        w = tokens[0]
        v = np.array(map(lambda x: float(x), tokens[1:]))
        word2vec_dict[w] = v
    return word2vec_dict, dim


def load_dataset():
    '''
    数据数据集
    :return:
    '''
    # 读取数据集
    train_X = pd.read_csv(train_x_pad_path, header=None).rename(columns={0: 'X'})
    train_Y = pd.read_csv(train_y_pad_path, header=None).rename(columns={0: 'Y'})
    test_X = pd.read_csv(test_x_pad_path, header=None).rename(columns={0: 'X'})
    # 保存词向量模型
    wv_model = Word2Vec.load(save_wv_model_path)
    return train_X['X'], train_Y['Y'], test_X['X'], wv_model


def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


# 加载停用词
stop_words = load_stop_words(stop_word_path)


def clean_sentence(sentence):
    '''
    特殊符号去除
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\!\/\|\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            '', sentence)
    else:
        return ' '


def filter_stopwords(words):
    '''
    过滤停用词
    :param seg_list: 切好词的列表 [word1 ,word2 .......]
    :return: 过滤后的停用词
    '''
    return [word for word in words if word not in stop_words]


def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    sentence = clean_sentence(sentence)
    # 切词，默认精确模式，全模式cut参数cut_all=True
    words = jieba.cut(sentence)
    # 过滤停用词
    words = filter_stopwords(words)
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


def sentences_proc(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc)
    return df


if __name__ == '__main__':
    # 数据集批量处理
    build_dataset(train_data_path, test_data_path)
