# -*- coding: utf-8 -*-

import os

import h5py
import torch
import torch.nn as nn


def init_embedding(embed):
    bias = (3. / embed.size(1)) ** 0.5
    nn.init.uniform_(embed, -bias, bias)


def get_sentences(fdata):
    start = 0
    sentences = []
    with open(fdata, 'r') as f:
        lines = [line for line in f]
    for i, line in enumerate(lines):
        if len(lines[i]) <= 1:
            splits = [l.split() for l in lines[start:i]]
            wordseq, tagseq = zip(*splits)
            start = i + 1
            while start < len(lines) and len(lines[start]) <= 1:
                start += 1
            sentences.append((wordseq, tagseq))

    return sentences


def get_elmo(fdata):
    felmo = os.path.splitext(fdata)[0] + '.hdf5'
    h5py_file = h5py.File(felmo, 'r')
    num_sentences = len(h5py_file.keys()) - 1
    reprs = [torch.tensor(h5py_file.get(str(i))).transpose(0, 1)
             for i in range(num_sentences)]

    return reprs


def get_parser(fdata):
    fparser = os.path.splitext(fdata)[0] + '.parser'

    reprs = torch.load(fparser)
    reprs = [torch.tensor(i) for i in reprs]

    return reprs


def get_embed(fembed):
    with open(fembed, 'r') as f:
        lines = [line for line in f]
    splits = [line.split() for line in lines]
    reprs = [
        (split[0], list(map(float, split[1:]))) for split in splits
    ]

    return reprs
