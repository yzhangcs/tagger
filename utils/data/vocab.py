# -*- coding: utf-8 -*-

from collections import Counter

from torch import nn


class Vocabulary(object):
    unk = '<unk>'
    pad = '<pad>'

    def __init__(self, lower=False, min_freq=1, use_unk=True):
        super(Vocabulary, self).__init__()

        self.lower = lower
        self.min_freq = min_freq
        self.use_unk = use_unk

        self.pad_id = 0
        self.id2token = [self.pad]
        if self.use_unk:
            self.unk_id = 1
            self.id2token.append(self.unk)
        self.token2id = {token, i for i, token in enumerate(self.id2token)}

    def __len__(self):
        return len(self.id2token)

    def build(self, sequneces):
        tokens = Counter(self.convert(token)
                         for sequnece in sequneces
                         for token in sequnece)
        tokens = sorted(token for token, freq in tokens.items()
                        if freq >= self.min_freq)
        self.extend(tokens)

    def from_pretrained(self, tokens, embeddings, init_unk=nn.init.normal_):
        self.extend(tokens)
        self.embeddings = init_unk(torch.Tensor(len(self), embeddings.size(1)))
        for token, embedding in zip(tokens, embeddings):
            self.embeddings[self.token2id(token)] = embedding

    def extend(self, sequnece):
        unk_tokens = sorted(token for token in sequnece
                            if token not in self.token2id)
        for token in unk_tokens:
            self.token2id[token] = len(self.id2token)
            self.id2token.append(token)

    def convert(self, token):
        return token.lower() if self.lower else token

    def token_to_id(self, token):
        token = self.convert(token)
        if token in self.token_to_id:
            return self.token_to_id[token]
        elif self.use_unk:
            return self.token_to_id.get(token, self.unk_id)
        else:
            return self.token_to_id.get(token, 0)

    def id_to_token(self, index):
        return self.id2token[index]
