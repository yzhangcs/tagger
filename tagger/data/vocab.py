# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Vocab(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, words, tags):
        # 填充索引
        self.pad_index = 0
        # 未知索引
        self.unk_index = 1

        self.words = [self.PAD, self.UNK] + sorted(words)
        self.chars = [self.PAD, self.UNK] + sorted(set(''.join(words)))
        self.tags = sorted(tags)

        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.cdict = {c: i for i, c in enumerate(self.chars)}
        self.tdict = {t: i for i, t in enumerate(self.tags)}

        self.n_words = len(self.words)
        self.n_chars = len(self.chars)
        self.n_tags = len(self.tags)

    def __repr__(self):
        info = f"{self.__class__.__name__}(\n"
        info += f"{'':2}num of words: {self.n_words}\n"
        info += f"{'':2}num of chars: {self.n_chars}\n"
        info += f"{'':2}num of tags: {self.n_tags}\n"
        info += f")"

        return info

    def word_to_id(self, sequence):
        ids = [self.wdict[w] if w in self.wdict
               else self.wdict.get(w.lower(), self.unk_index)
               for w in sequence]
        ids = torch.tensor(ids, dtype=torch.long)

        return ids

    def char_to_id(self, sequence, fix_length=20):
        char_ids = torch.zeros(len(sequence), fix_length, dtype=torch.long)
        for i, word in enumerate(sequence):
            ids = torch.tensor([self.cdict.get(c, self.unk_index)
                                for c in word[:fix_length]], dtype=torch.long)
            char_ids[i, :len(ids)] = ids

        return char_ids

    def tag_to_id(self, sequence):
        ids = [self.tdict.get(token, 0) for token in sequence]
        ids = torch.tensor(ids, dtype=torch.long)

        return ids

    def read_embeddings(self, embed, unk=None, init_unk=nn.init.normal_):
        words = embed.words
        if unk:
            words[words.index(unk)] = self.UNK

        self.extend(words)
        self.embeddings = torch.Tensor(self.n_words, embed.dim)

        for i, word in enumerate(self.words):
            if word in embed:
                self.embeddings[i] = embed[word]
            elif word.lower() in embed:
                self.embeddings[i] = embed[word.lower()]
            else:
                init_unk(self.embeddings[i])

    def extend(self, words):
        # 扩展词汇和字符
        self.words.extend({w for w in words if w not in self.wdict})
        self.chars.extend({c for c in ''.join(words) if c not in self.cdict})
        # 更新字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.cdict = {c: i for i, c in enumerate(self.chars)}
        # 更新词汇和字符数
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    @classmethod
    def from_corpus(cls, corpus, min_freq=1):
        words = list(w for w, f in corpus.words.items() if f >= min_freq)
        tags = list(corpus.tags)
        vocab = cls(words, tags)

        return vocab

    @classmethod
    def load(cls, filename):
        return torch.load(filename)

    def save(filename):
        torch.save(self, filename)
