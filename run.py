# -*- coding: utf-8 -*-

import argparse

import torch
from torch import nn

import config
import utils
from corpus import Corpus
from dataset import TextDataset
from models import CHAR_LSTM_CRF, ELMO_LSTM_CRF, PARSER_LSTM_CRF
from trainer import Trainer

if __name__ == '__main__':
    # 解析命令参数
    parser = argparse.ArgumentParser(
        description='Create several models for Sequence Labeling.'
    )
    parser.add_argument('--model', default='char_lstm_crf',
                        choices=['char_lstm_crf',
                                 'elmo_lstm_crf',
                                 'parser_lstm_crf'],
                        help='choose the model for Sequence Labeling')
    parser.add_argument('--task',  default='ner',
                        choices=['chunking', 'ner', 'pos'],
                        help='choose the model for Sequence Labeling')
    parser.add_argument('--crf', action='store_true', default=False,
                        help='use crf')
    parser.add_argument('--drop', action='store', default=0.5, type=float,
                        help='set the prob of dropout')
    parser.add_argument('--batch_size', action='store', default=50, type=int,
                        help='set the size of batch')
    parser.add_argument('--epochs', action='store', default=100, type=int,
                        help='set the max num of epochs')
    parser.add_argument('--interval', action='store', default=10, type=int,
                        help='set the num of epochs to be patient before early stopping')
    parser.add_argument('--lr', action='store', default=0.001, type=float,
                        help='set the learning rate of training')
    parser.add_argument('--threads', '-t', action='store', default=4, type=int,
                        help='set the max num of threads')
    parser.add_argument('--device', '-d', action='store', default=-1, type=int,
                        help='set the id of GPU to use')
    parser.add_argument('--seed', '-s', action='store', default=1, type=int,
                        help='set the seed for generating random numbers')
    parser.add_argument('--file', '-f', action='store', default='model.pt',
                        help='set where to store the model')
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    if args.device >= 0:
        torch.cuda.set_device(args.device)

    # 根据模型读取配置
    config = config.config[args.model]

    print("Preprocess the data")
    # 建立语料
    corpus = Corpus(config.ftrain, config.fembed)
    print(corpus)

    print("Load the dataset")
    trainset = TextDataset(fdata=config.ftrain,
                           corpus=corpus,
                           use_char=config.use_char,
                           use_elmo=config.use_elmo,
                           use_parser=config.use_parser)
    devset = TextDataset(fdata=config.fdev,
                         corpus=corpus,
                         use_char=config.use_char,
                         use_elmo=config.use_elmo,
                         use_parser=config.use_parser)
    testset = TextDataset(fdata=config.ftest,
                          corpus=corpus,
                          use_char=config.use_char,
                          use_elmo=config.use_elmo,
                          use_parser=config.use_parser)
    print(f"{'':2}size of trainset: {len(trainset)}")
    print(f"{'':2}size of devset: {len(devset)}")
    print(f"{'':2}size of testset: {len(testset)}")

    # 设置随机数种子
    torch.manual_seed(args.seed)

    print("Create Neural Network")
    if args.model == 'char_lstm_crf':
        print(f"{'':2}n_char: {corpus.n_chars}\n"
              f"{'':2}n_char_embed: {config.n_char_embed}\n"
              f"{'':2}n_char_out: {config.n_char_out}\n"
              f"{'':2}n_vocab: {corpus.n_words}\n"
              f"{'':2}n_embed: {config.n_embed}\n"
              f"{'':2}n_hidden: {config.n_hidden}\n"
              f"{'':2}n_out: {corpus.n_tags}\n")
        model = CHAR_LSTM_CRF(n_char=corpus.n_chars,
                              n_char_embed=config.n_char_embed,
                              n_char_out=config.n_char_out,
                              n_vocab=corpus.n_words,
                              n_embed=config.n_embed,
                              n_hidden=config.n_hidden,
                              n_out=corpus.n_tags,
                              drop=args.drop)
    elif args.model == 'elmo_lstm_crf':
        print(f"{'':2}n_elmo: {config.n_elmo}\n"
              f"{'':2}n_vocab: {corpus.n_words}\n"
              f"{'':2}n_embed: {config.n_embed}\n"
              f"{'':2}n_hidden: {config.n_hidden}\n"
              f"{'':2}n_out: {corpus.n_tags}\n")
        model = ELMO_LSTM_CRF(n_elmo=config.n_elmo,
                              n_vocab=corpus.n_words,
                              n_embed=config.n_embed,
                              n_hidden=config.n_hidden,
                              n_out=corpus.n_tags,
                              drop=args.drop)
    elif args.model == 'parser_lstm_crf':
        print(f"{'':2}n_char: {corpus.n_chars}\n"
              f"{'':2}n_char_embed: {config.n_char_embed}\n"
              f"{'':2}n_char_out: {config.n_char_out}\n"
              f"{'':2}n_parser: {config.n_parser}\n"
              f"{'':2}n_vocab: {corpus.n_words}\n"
              f"{'':2}n_embed: {config.n_embed}\n"
              f"{'':2}n_hidden: {config.n_hidden}\n"
              f"{'':2}n_out: {corpus.n_tags}\n")
        model = PARSER_LSTM_CRF(n_char=corpus.n_chars,
                                n_char_embed=config.n_char_embed,
                                n_char_out=config.n_char_out,
                                n_parser=config.n_parser,
                                n_vocab=corpus.n_words,
                                n_embed=config.n_embed,
                                n_hidden=config.n_hidden,
                                n_out=corpus.n_tags,
                                drop=args.drop)
    model.load_pretrained(corpus.embed)
    if args.device >= 0:
        model = model.cuda()
    print(f"{model}\n")

    trainer = Trainer(model=model, corpus=corpus, task=args.task)
    trainer.fit(trainset=trainset,
                devset=devset,
                testset=testset,
                batch_size=args.batch_size,
                epochs=args.epochs,
                interval=args.interval,
                lr=args.lr,
                file=args.file)
