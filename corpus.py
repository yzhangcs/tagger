# -*- coding: utf-8 -*-

from collections import Counter

import torch
import torch.nn as nn

from utils import init_embedding, get_sentences, get_embed


class Corpus(object):
    PAD = '<PAD>'
    UNK = '<UNK>'

    def __init__(self, fdata, fembed):
        # 获取数据的句子、词汇、词性和字符
        self.sents, self.words, self.tags, self.chars = self.parse_sents(fdata)
        # 增加填充词汇和未知词汇
        self.words = [self.PAD, self.UNK] + self.words
        # 增加填充字符和未知字符
        self.chars = [self.PAD, self.UNK] + self.chars

        # 词汇字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        # 词性字典
        self.tdict = {t: i for i, t in enumerate(self.tags)}
        # 字符字典
        self.cdict = {c: i for i, c in enumerate(self.chars)}

        # 填充词汇索引
        self.pad_wi = self.wdict[self.PAD]
        # 未知词汇索引
        self.unk_wi = self.wdict[self.UNK]
        # 填充字符索引
        self.pad_ci = self.cdict[self.PAD]
        # 未知字符索引
        self.unk_ci = self.cdict[self.UNK]

        # 句子数量
        self.n_sents = len(self.sents)
        # 词汇数量
        self.n_words = len(self.words)
        # 词性数量
        self.n_tags = len(self.tags)
        # 字符数量
        self.n_chars = len(self.chars)

        # 预训练词嵌入
        self.embed = self.parse_embed(fembed)

    def __repr__(self):
        info = f"{self.__class__.__name__}(\n"
        info += f"{'':2}num of sentences: {self.n_sents}\n"
        info += f"{'':2}num of words: {self.n_words}\n"
        info += f"{'':2}num of tags: {self.n_tags}\n"
        info += f"{'':2}num of chars: {self.n_chars}\n"
        info += f")\n"

        return info

    def extend(self, words):
        unk_words = [w for w in words if w not in self.wdict]
        unk_chars = [c for c in ''.join(unk_words) if c not in self.cdict]
        # 扩展词汇和字符
        self.words = sorted(set(self.words + unk_words) - {self.PAD})
        self.chars = sorted(set(self.chars + unk_chars) - {self.PAD})
        self.words = [self.PAD] + self.words
        self.chars = [self.PAD] + self.chars
        # 更新字典
        self.wdict = {w: i for i, w in enumerate(self.words)}
        self.cdict = {c: i for i, c in enumerate(self.chars)}
        # 更新索引
        self.pad_wi = self.wdict[self.PAD]
        self.unk_wi = self.wdict[self.UNK]
        self.pad_ci = self.cdict[self.PAD]
        self.unk_ci = self.cdict[self.UNK]
        # 更新词汇和字符数
        self.n_words = len(self.words)
        self.n_chars = len(self.chars)

    def load(self, fdata, use_char=False, max_len=20):
        sents = get_sentences(fdata)
        x, y, char_x = [], [], []

        for wordseq, tagseq in sents:
            wiseq = [self.wdict[w] if w in self.wdict
                     else self.wdict.get(w.lower(), self.unk_wi)
                     for w in wordseq]
            tiseq = [self.tdict.get(t, 0) for t in tagseq]
            x.append(torch.tensor(wiseq, dtype=torch.long).cuda())
            y.append(torch.tensor(tiseq, dtype=torch.long).cuda())
            char_x.append(torch.tensor([
                [self.cdict.get(c, self.unk_ci) for c in w[:max_len]] +
                [0] * (max_len - len(w))
                for w in wordseq
            ]).cuda())
        reprs = (x, y) if not use_char else (x, y, char_x)

        return reprs

    def parse_sents(self, fdata, threshold=1):
        sents = get_sentences(fdata)
        wordseqs, tagseqs = zip(*sents)
        words = Counter(w for wordseq in wordseqs for w in wordseq)
        words = sorted(w for w, f in words.items() if f > threshold)
        tags = sorted(set(t for tagseq in tagseqs for t in tagseq))
        chars = sorted(set(''.join(words)))

        return sents, words, tags, chars

    def parse_embed(self, fembed):
        words, embed = zip(*get_embed(fembed))
        # 扩充词汇
        self.extend(words)
        # 初始化词嵌入
        embed = torch.tensor(embed, dtype=torch.float)
        embed_indices = [self.wdict[w] for w in words]
        upper_indices = [self.wdict[w] for w in self.words if not w.islower()]
        lower_indices = [self.wdict.get(w.lower(), self.unk_wi)
                         for w in self.words if not w.islower()]
        extended_embed = torch.Tensor(self.n_words, embed.size(1))
        init_embedding(extended_embed)

        extended_embed[embed_indices] = embed
        # 非小写单词的词嵌入用小写单词的词嵌入代替
        extended_embed[upper_indices] = extended_embed[lower_indices]

        return extended_embed
