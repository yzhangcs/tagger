# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim

from metric import AccuracyMethod, SpanF1Method


class Trainer(object):

    def __init__(self, model, corpus, task):
        super(Trainer, self).__init__()

        self.model = model
        self.corpus = corpus
        self.task = task

    def train(self, train_loader, dev_loader, test_loader,
              epochs, patience, lr, file):
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大度量及对应的迭代次数
        max_e, max_metric = 0, 0.0
        # 设置优化器为Adam
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            start = datetime.now()
            # 更新参数
            self.train(train_loader)

            print(f"Epoch: {epoch} / {epochs}:")
            loss, train_metric = self.evaluate(train_loader)
            print(f"{'train:':<6} Loss: {loss:.4f} {train_metric}")
            loss, dev_metric = self.evaluate(dev_loader)
            print(f"{'dev:':<6} Loss: {loss:.4f} {dev_metric}")
            loss, test_metric = self.evaluate(test_loader)
            print(f"{'test:':<6} Loss: {loss:.4f} {test_metric}")
            t = datetime.now() - start
            print(f"{t}s elapsed\n")
            total_time += t

            # 保存效果最好的模型
            if dev_metric > max_metric:
                torch.save(self.model, file)
                max_e, max_metric = epoch, dev_metric
            elif epoch - max_e >= patience:
                break
        self.model = torch.load(file).cuda()
        loss, metric = self.evaluate(test_loader)

        print(f"max score of dev is {max_metric.score:.2%} at epoch {max_e}")
        print(f"the score of test at epoch {max_e} is {metric.score:.2%}")
        print(f"mean time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

    def train(self, loader):
        # 设置为训练模式
        self.model.train()

        # 从加载器中加载数据进行训练
        for x, char_x, y in loader:
            # 清除梯度
            self.optimizer.zero_grad()
            # 获取掩码
            mask = x.gt(0)

            out = self.model(x, char_x)
            out = out.transpose(0, 1)  # [T, B, N]
            y, mask = y.t(), mask.t()  # [T, B]
            loss = self.model.crf(out, y, mask)
            # 计算梯度
            loss.backward()
            # 限制梯度范围
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            # 更新参数
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader):
        # 设置为评价模式
        self.model.eval()

        loss = 0
        if self.task == 'pos':
            metric = AccuracyMethod()
        else:
            metric = SpanF1Method(self.corpus.tags)

        # 从加载器中加载数据进行评价
        for x, char_x, y in loader:
            mask = x.gt(0)
            lens = mask.sum(dim=1)
            targets = torch.split(y[mask], lens.tolist())

            out = self.model(x, char_x)
            out = out.transpose(0, 1)  # [T, B, N]
            y, mask = y.t(), mask.t()  # [T, B]
            predicts = self.model.crf.viterbi(out, mask)
            loss += self.model.crf(out, y, mask)

            metric(predicts, targets)
        loss /= len(loader)

        return loss, metric
