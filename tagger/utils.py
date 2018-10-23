# -*- coding: utf-8 -*-

import os

import h5py
import torch
import torch.nn as nn


def init_embedding(embed):
    bias = (3. / embed.size(-1)) ** 0.5
    nn.init.uniform_(embed, -bias, bias)


def get_elmo(fdata):
    felmo = os.path.splitext(fdata)[0] + '.hdf5'
    h5py_file = h5py.File(felmo, 'r')
    num_sentences = len(h5py_file.keys()) - 1
    reprs = [torch.tensor(h5py_file.get(str(i))).transpose(0, 1)
             for i in range(num_sentences)]

    return reprs


def numericalize(vocab, corpus, use_char, use_elmo):
    x = [vocab.word_to_id(seq) for seq in corpus.x_seqs]
    items = [x]
    if use_char:
        items.append([vocab.char_to_id(seq) for seq in corpus.x_seqs])
    if use_elmo:
        items.append(get_elmo(corpus.filename))
    y = [vocab.tag_to_id(seq) for seq in corpus.y_seqs]
    items.append(y)

    return items
