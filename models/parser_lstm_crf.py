# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

from modules import CRF, CharLSTM, ScalarMix


class PARSER_LSTM_CRF(nn.Module):

    def __init__(self, n_char, n_char_embed, n_char_out,
                 n_parser, n_vocab, n_embed, n_hidden, n_out, drop=0.5):
        super(PARSER_LSTM_CRF, self).__init__()

        self.embed = nn.Embedding(n_vocab, n_embed)
        self.scalar_mix = ScalarMix(n_reprs=3, do_layer_norm=False)

        # 字嵌入LSTM层
        self.char_lstm = CharLSTM(n_char=n_char,
                                  n_embed=n_char_embed,
                                  n_out=n_char_out)
        # 词嵌入LSTM层
        self.word_lstm = nn.LSTM(input_size=n_embed + n_char_out + n_parser,
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

    def load_pretrained(self, embed):
        self.embed = nn.Embedding.from_pretrained(embed, False)

    def forward(self, x, char_x, parser):
        B, T = x.shape
        # 获取掩码
        mask = x.gt(0)
        # 获取句子长度
        lens = mask.sum(dim=1)
        # 获取词嵌入向量
        x = self.embed(x)
        # 获取字嵌入向量
        char_x = self.char_lstm(char_x[mask])
        char_x = pad_sequence(torch.split(char_x, lens.tolist()), True)
        # 获取parser
        parser = self.scalar_mix(parser.permute(2, 0, 1, 3))

        # 获取词表示与parser的拼接
        x = torch.cat((x, char_x, parser), dim=-1)
        x = self.drop(x)

        x = pack_padded_sequence(x, lens, True)
        x, _ = self.word_lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.hid(x)

        return self.out(x)

    def collate_fn(self, data):
        x, y, char_x, parser = zip(
            *sorted(data, key=lambda x: len(x[0]), reverse=True)
        )
        x = pad_sequence(x, True).cuda()
        y = pad_sequence(y, True).cuda()
        char_x = pad_sequence(char_x, True).cuda()
        parser = pad_sequence(parser, True).cuda()

        return x, y, char_x, parser
