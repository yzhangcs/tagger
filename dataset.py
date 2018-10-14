# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

from utils import get_elmo, get_parser


class TextDataset(Dataset):

    def __init__(self, fdata, corpus, use_char, use_elmo, use_parser):
        super(TextDataset, self).__init__()

        self.fdata = fdata
        self.corpus = corpus
        self.items = self.get_items(use_char, use_elmo, use_parser)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def get_items(self, use_char, use_elmo, use_parser):
        reprs = self.corpus.load(self.fdata, use_char)
        if use_elmo:
            reprs.append(get_elmo(self.fdata))
        if use_parser:
            reprs.append(get_parser(self.fdata))
        items = list(zip(*reprs))

        return items
