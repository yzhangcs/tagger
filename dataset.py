# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

from utils import get_elmo


class TextDataset(Dataset):

    def __init__(self, fdata, corpus, use_char=False, use_elmo=False):
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
        if not use_elmo:
            items = list(zip(*reprs))
        else:
            items = list(zip(*reprs, get_elmo(self.fdata)))

        return items
