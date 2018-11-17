# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CRF(nn.Module):

    def __init__(self, n_tags, batch_first=True):
        super(CRF, self).__init__()

        self.n_tags = n_tags
        self.batch_first = batch_first
        self.trans = nn.Parameter(torch.Tensor(n_tags, n_tags))  # (FROM->TO)
        self.strans = nn.Parameter(torch.Tensor(n_tags))
        self.etrans = nn.Parameter(torch.Tensor(n_tags))

        self.reset_parameters()

    def extra_repr(self):
        info = f"n_tags={self.n_tags}"
        if self.batch_first:
            info += f", batch_first={self.batch_first}"

        return info

    def reset_parameters(self):
        nn.init.zeros_(self.trans)
        nn.init.zeros_(self.strans)
        nn.init.zeros_(self.etrans)

    def forward(self, emit, target, mask):
        logZ = self.get_logZ(emit, mask)
        score = self.get_score(emit, target, mask)

        return logZ - score

    def get_logZ(self, emit, mask):
        if self.batch_first:
            emit, mask = emit.transpose(0, 1), mask.t()
        T, B, N = emit.shape

        alpha = self.strans + emit[0]  # [B, N]

        for i in range(1, T):
            trans_i = self.trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            mask_i = mask[i].unsqueeze(1).expand_as(alpha)  # [B, N]
            scores = trans_i + emit_i + alpha.unsqueeze(2)  # [B, N, N]
            scores = torch.logsumexp(scores, dim=1)  # [B, N]
            alpha[mask_i] = scores[mask_i]
        logZ = torch.logsumexp(alpha + self.etrans, dim=1).sum()

        return logZ / B

    def get_score(self, emit, target, mask):
        if self.batch_first:
            emit, target, mask = emit.transpose(0, 1), target.t(), mask.t()
        T, B, N = emit.shape
        scores = emit.new_zeros(T, B)

        # plus the transition score
        scores[1:] += self.trans[target[:-1], target[1:]]
        # plus the emit score
        scores += emit.gather(dim=2, index=target.unsqueeze(2)).squeeze(2)
        # filter some scores by mask
        score = scores.masked_select(mask).sum()

        ends = mask.sum(dim=0).view(1, -1) - 1
        # plus the score of start transitions
        score += self.strans[target[0]].sum()
        # plus the score of end transitions
        score += self.etrans[target.gather(dim=0, index=ends)].sum()

        return score / B

    def viterbi(self, emit, mask):
        if self.batch_first:
            emit, mask = emit.transpose(0, 1), mask.t()
        T, B, N = emit.shape
        lens = mask.sum(dim=0)
        delta = emit.new_zeros(T, B, N)
        paths = emit.new_zeros(T, B, N, dtype=torch.long)

        delta[0] = self.strans + emit[0]  # [B, N]

        for i in range(1, T):
            trans_i = self.trans.unsqueeze(0)  # [1, N, N]
            emit_i = emit[i].unsqueeze(1)  # [B, 1, N]
            scores = trans_i + emit_i + delta[i - 1].unsqueeze(2)  # [B, N, N]
            delta[i], paths[i] = torch.max(scores, dim=1)

        predicts = []
        for i, length in enumerate(lens):
            prev = torch.argmax(delta[length - 1, i] + self.etrans)

            predict = [prev]
            for j in reversed(range(1, length)):
                prev = paths[j, i, prev]
                predict.append(prev)
            # flip the predicted sequence before appending it to the list
            predicts.append(paths.new_tensor(predict).flip(0))

        return predicts
