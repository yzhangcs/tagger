# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from metric import AccuracyMethod, SpanF1Method


class Trainer(object):

    def __init__(self, model, corpus, task):
        super(Trainer, self).__init__()

        self.model = model
        self.corpus = corpus
        self.task = task

    def fit(self, trainset, devset, testset,
            batch_size, epochs, interval, lr, file):
        # 设置数据加载器
        train_loader = DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=self.model.collate_fn)
        dev_loader = DataLoader(dataset=devset,
                                batch_size=batch_size,
                                collate_fn=self.model.collate_fn)
        test_loader = DataLoader(dataset=testset,
                                 batch_size=batch_size,
                                 collate_fn=self.model.collate_fn)
        # 记录迭代时间
        total_time = timedelta()
        # 记录最大准确率及对应的迭代次数
        max_e, max_val = 0, 0.0
        # 设置优化器为Adam
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)

        if self.task == 'pos':
            for epoch in range(1, epochs + 1):
                start = datetime.now()
                # 更新参数
                self.train(train_loader)

                print(f"Epoch: {epoch} / {epochs}:")
                loss, train_acc = self.evaluate(train_loader)
                print(f"{'train:':<6} "
                      f"Loss: {loss:.4f} "
                      f"Accuracy: {train_acc:.2%}")
                loss, dev_acc = self.evaluate(dev_loader)
                print(f"{'dev:':<6} "
                      f"Loss: {loss:.4f} "
                      f"Accuracy: {dev_acc:.2%}")
                loss, test_acc = self.evaluate(test_loader)
                print(f"{'test:':<6} "
                      f"Loss: {loss:.4f} "
                      f"Accuracy: {test_acc:.2%}")
                t = datetime.now() - start
                print(f"{t}s elapsed\n")
                total_time += t

                # 保存效果最好的模型
                if dev_acc > max_val:
                    torch.save(self.model, file)
                    max_e, max_val = epoch, dev_acc
                elif epoch - max_e >= interval:
                    break
            self.model = torch.load(file)
            loss, test_acc = self.evaluate(test_loader)

            print(f"max accuracy of dev is {max_val:.2%} at epoch {max_e}")
            print(f"the accuracy of test at epoch {max_e} is {test_acc:.2%}")
            print(f"mean time of each epoch is {total_time / epoch}s")
            print(f"{total_time}s elapsed")
        else:
            for epoch in range(1, epochs + 1):
                start = datetime.now()
                # 更新参数
                self.train(train_loader)

                print(f"Epoch: {epoch} / {epochs}:")
                loss, p, r, train_f = self.evaluate(train_loader)
                print(f"{'train:':<6} "
                      f"Loss: {loss:.4f} "
                      f"Precision: {p:.2%} "
                      f"Recall: {r:.2%} "
                      f"F: {train_f:.2%}")
                loss, p, r, dev_f = self.evaluate(dev_loader)
                print(f"{'dev:':<6} "
                      f"Loss: {loss:.4f} "
                      f"Precision: {p:.2%} "
                      f"Recall: {r:.2%} "
                      f"F: {dev_f:.2%}")
                loss, p, r, test_f = self.evaluate(test_loader)
                print(f"{'test:':<6} "
                      f"Loss: {loss:.4f} "
                      f"Precision: {p:.2%} "
                      f"Recall: {r:.2%} "
                      f"F: {test_f:.2%}")
                t = datetime.now() - start
                print(f"{t}s elapsed\n")
                total_time += t

                # 保存效果最好的模型
                if dev_f > max_val:
                    torch.save(self.model, file)
                    max_e, max_val = epoch, dev_f
                elif epoch - max_e >= interval:
                    break
            self.model = torch.load(file)
            loss, p, r, test_f = self.evaluate(test_loader)

            print(f"max F-value of dev is {max_val:.2%} at epoch {max_e}")
            print(f"the F-value of test at epoch {max_e} is {test_f:.2%}")
            print(f"mean time of each epoch is {total_time / epoch}s")
            print(f"{total_time}s elapsed")

    def train(self, loader):
        # 设置为训练模式
        self.model.train()

        # 从加载器中加载数据进行训练
        for x, y, char_x in loader:
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
        for x, y, char_x in loader:
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

        if self.task == 'pos':
            return (loss, metric.accuracy)
        else:
            return (loss, metric.precision, metric.recall, metric.f_score)
