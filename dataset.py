# -*- coding: utf-8 -*-

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import get_elmo


def collate_fn(data):
    reprs = zip(
        *sorted(data, key=lambda x: len(x[0]), reverse=True)
    )
    reprs = (pad_sequence(i, True) for i in reprs)

    if torch.cuda.is_available():
        reprs = (i.cuda() for i in reprs)

    return reprs


class TextDataset(Dataset):

    def __init__(self, fdata, corpus, use_char, use_elmo):
        super(TextDataset, self).__init__()

        self.fdata = fdata
        self.corpus = corpus
        self.items = self.get_items(use_char, use_elmo)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def get_items(self, use_char, use_elmo):
        reprs = self.corpus.load(self.fdata, use_char)
        if use_elmo:
            reprs.append(get_elmo(self.fdata))
        items = list(zip(*reprs))

        return items
