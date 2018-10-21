# -*- coding: utf-8 -*-

from collections import Counter

import torch
import torch.nn as nn

from utils import get_embed, get_sentences, init_embedding


class Corpus(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, fdata, fembed):
        # 填充索引
        self.pad_index = 0
        # 未知索引
        self.unk_index = 1

        # 获取数据的词汇、字符和词性
        self.words, self.chars, self.tags = self.parse_sents(fdata)
        # 增加填充词汇和未知词汇
        self.words = [self.PAD, self.UNK] + self.words
        # 增加填充字符和未知字符
        self.chars = [self.PAD, self.UNK] + self.chars

        # 词汇字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        # 字符字典
        self.cdict = {c: i for i, c in enumerate(self.chars)}
        # 词性字典
        self.tdict = {t: i for i, t in enumerate(self.tags)}

        # 预训练词嵌入
        self.embed = self.parse_embed(fembed)

        # 词汇数量
        self.n_words = len(self.words)
        # 字符数量
        self.n_chars = len(self.chars)
        # 词性数量
        self.n_tags = len(self.tags)

    def __repr__(self):
        info = f"{self.__class__.__name__}(\n"
        info += f"{'':2}num of words: {self.n_words}\n"
        info += f"{'':2}num of chars: {self.n_chars}\n"
        info += f"{'':2}num of tags: {self.n_tags}\n"
        info += f")"

        return info

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

    def load(self, fdata, use_char=False, max_len=20):
        sentences = get_sentences(fdata)
        x, char_x, y = [], [], []

        for wordseq, tagseq in sentences:
            wiseq = [self.wdict[w] if w in self.wdict
                     else self.wdict.get(w.lower(), self.unk_index)
                     for w in wordseq]
            tiseq = [self.tdict.get(t, 0) for t in tagseq]
            x.append(torch.tensor(wiseq, dtype=torch.long))
            char_x.append(torch.tensor([
                [self.cdict.get(c, self.unk_index) for c in w[:max_len]] +
                [0] * (max_len - len(w))
                for w in wordseq
            ]))
            y.append(torch.tensor(tiseq, dtype=torch.long))
        inputs = [x] if not use_char else [x, char_x]

        return inputs, y

    def parse_sents(self, fdata, min_freq=1):
        sentences = get_sentences(fdata)
        wordseqs, tagseqs = zip(*sentences)
        words = Counter(w for wordseq in wordseqs for w in wordseq)
        words = sorted(w for w, f in words.items() if f > min_freq)
        chars = sorted(set(''.join(words)))
        tags = sorted(set(t for tagseq in tagseqs for t in tagseq))

        return words, chars, tags

    def parse_embed(self, fembed, unk='unk'):
        words, embed = zip(*get_embed(fembed))
        words = [w if w != unk else self.UNK for w in words]
        # 扩充词汇
        self.extend(words)
        # 初始化词嵌入
        embed = torch.tensor(embed, dtype=torch.float)
        embed_indices = [self.wdict[w] for w in words]
        upper_indices = [self.wdict[w] for w in self.words if not w.islower()]
        lower_indices = [self.wdict.get(w.lower(), self.unk_index)
                         for w in self.words if not w.islower()]
        extended_embed = torch.Tensor(self.n_words, embed.size(1))
        init_embedding(extended_embed)

        extended_embed[embed_indices] = embed
        # 非小写单词的词嵌入用小写单词的词嵌入代替
        extended_embed[upper_indices] = extended_embed[lower_indices]

        return extended_embed
