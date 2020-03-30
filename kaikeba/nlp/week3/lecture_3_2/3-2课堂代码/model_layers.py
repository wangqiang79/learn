# -*- coding:utf-8 -*-
# Created by LuoJie at 11/23/19
import tensorflow as tf

from utils.config import save_wv_model_path
from utils.wv_loader import get_vocab, load_word2vec_file
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy


def seq2seq(input_length, output_sequence_length, embedding_matrix, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], trainable=False,
                        input_length=input_length))
    model.add(Bidirectional(GRU(300, return_sequences=False)))
    model.add(Dense(300, activation="relu"))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(300, return_sequences=True)))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(1e-3))
    model.summary()
    return model
