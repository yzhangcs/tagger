# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

from tagger.modules import CRF, ScalarMix


class ELMO_LSTM_CRF(nn.Module):

    def __init__(self, n_vocab, n_embed, n_elmo, n_hidden, n_out, drop=0.5):
        super(ELMO_LSTM_CRF, self).__init__()

        self.embed = nn.Embedding(n_vocab, n_embed)
        self.scalar_mix = ScalarMix(n_reprs=3, do_layer_norm=False)

        # the word-lstm layer
        self.lstm = nn.LSTM(input_size=n_embed + n_elmo,
                            hidden_size=n_hidden,
                            batch_first=True,
                            bidirectional=True)

        # the hidden layer
        self.hid = nn.Linear(n_hidden * 2, n_hidden)
        self.activation = nn.Tanh()
        # the output layer
        self.out = nn.Linear(n_hidden, n_out)
        # the CRF layer
        self.crf = CRF(n_out)

        self.drop = nn.Dropout(drop)

        self.reset_parameters()

    def reset_parameters(self):
        # init Linear
        nn.init.xavier_uniform_(self.hid.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def load_pretrained(self, embed):
        self.embed = nn.Embedding.from_pretrained(embed, False)

    def forward(self, x, elmo):
        # get the mask and lengths of given batch
        mask = x.gt(0)
        lens = mask.sum(dim=1)
        # get embeddings and elmo
        x = self.embed(x[mask])
        elmo = self.scalar_mix(elmo[mask].transpose(0, 1))

        # concatenate all the representations
        x = torch.cat((x, elmo), dim=-1)
        x = self.drop(x)

        x = pack_sequence(torch.split(x, lens.tolist()))
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.hid(x)
        x = self.activation(x)

        return self.out(x)
