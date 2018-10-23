# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from tagger.modules import CRF, CharLSTM


class CHAR_LSTM_CRF(nn.Module):

    def __init__(self, n_vocab, n_embed,
                 n_char, n_char_embed, n_char_out,
                 n_hidden, n_out, drop=0.5):
        super(CHAR_LSTM_CRF, self).__init__()

        self.embed = nn.Embedding(n_vocab, n_embed)
        # 字嵌入LSTM层
        self.char_lstm = CharLSTM(n_char=n_char,
                                  n_embed=n_char_embed,
                                  n_out=n_char_out)

        # 词嵌入LSTM层
        self.word_lstm = nn.LSTM(input_size=n_embed + n_char_out,
                                 hidden_size=n_hidden,
                                 batch_first=True,
                                 bidirectional=True)

        # 隐藏层
        self.hid = nn.Sequential(nn.Linear(n_hidden * 2, n_hidden), nn.Tanh())
        # 输出层
        self.out = nn.Linear(n_hidden, n_out)
        # CRF层
        self.crf = CRF(n_out)

        self.drop = nn.Dropout(drop)

        # 初始化权重
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.Embedding:
            bias = (3. / m.weight.size(1)) ** 0.5
            nn.init.uniform_(m.weight, -bias, bias)

    def load_pretrained(self, embed):
        self.embed = nn.Embedding.from_pretrained(embed, False)

    def forward(self, x, char_x):
        # 获取掩码
        mask = x.gt(0)
        # 获取句子长度
        lens = mask.sum(dim=1)
        # 获取词嵌入向量
        x = self.embed(x[mask])
        # 获取字嵌入向量
        char_x = self.char_lstm(char_x[mask])

        # 获取词表示与字表示的拼接
        x = torch.cat((x, char_x), dim=-1)
        x = self.drop(x)

        x = pack_sequence(torch.split(x, lens.tolist()))
        x, _ = self.word_lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.hid(x)

        return self.out(x)