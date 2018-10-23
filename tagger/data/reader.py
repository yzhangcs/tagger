# -*- coding: utf-8 -*-

from collections import Counter

import torch


class Corpus(object):

    def __init__(self, filename):
        super(Corpus, self).__init__()

        self.filename = filename
        self.x_seqs, self.y_seqs = self.read(filename)

    @property
    def words(self):
        return Counter(w for seq in self.x_seqs for w in seq)

    @property
    def tags(self):
        return Counter(t for seq in self.y_seqs for t in seq)

    @classmethod
    def read(cls, filename):
        start = 0
        x_seqs, y_seqs = [], []
        with open(filename, 'r') as f:
            lines = [line for line in f]
        for i, line in enumerate(lines):
            if len(lines[i]) <= 1:
                x_seq, y_seq = zip(*[l.split() for l in lines[start:i]])
                start = i + 1
                while start < len(lines) and len(lines[start]) <= 1:
                    start += 1
                x_seqs.append(list(x_seq))
                y_seqs.append(list(y_seq))

        return x_seqs, y_seqs


class Embedding(object):

    def __init__(self, filename):
        super(Embedding, self).__init__()

        self.filename = filename
        self.words, self.vectors = self.read(filename)
        self.pretrained = {w: v for w, v in zip(self.words, self.vectors)}

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return torch.tensor(self.pretrained[word], dtype=torch.float)

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def read(cls, filename):
        with open(filename, 'r') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        reprs = [
            (split[0], list(map(float, split[1:]))) for split in splits
        ]
        words, vectors = map(list, zip(*reprs))

        return words, vectors
