# -*- coding:utf-8 -*-
# Created by LuoJie at 11/22/19

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def tokenize(lines, max_len, num_words=500):
    tokenizer = Tokenizer(filters='', num_words=num_words)
    tokenizer.fit_on_texts(lines)
    word_index = tokenizer.word_index
    tensor = tokenizer.texts_to_sequences(lines)
    tensor = pad_sequences.pad_sequences(tensor, padding='post', max_len=max_len)
    return tensor, tokenizer, word_index
