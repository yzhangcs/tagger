# -*- coding: utf-8 -*-

import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from tagger import Trainer
from tagger.data import Corpus, Embedding, TextDataset, Vocab, collate_fn
from tagger.models import CHAR_LSTM_CRF, ELMO_LSTM_CRF
from tagger.utils import init_embedding, numericalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create several models for Sequence Labeling.'
    )
    parser.add_argument('--model', default='char_lstm_crf',
                        choices=['char_lstm_crf', 'elmo_lstm_crf'],
                        help='choose the model for Sequence Labeling')
    parser.add_argument('--task',  default='ner',
                        choices=['chunking', 'ner', 'pos'],
                        help='choose the task of Sequence Labeling')
    parser.add_argument('--drop', action='store', default=0.5, type=float,
                        help='set the prob of dropout')
    parser.add_argument('--batch_size', action='store', default=50, type=int,
                        help='set the size of batch')
    parser.add_argument('--epochs', action='store', default=100, type=int,
                        help='set the max num of epochs')
    parser.add_argument('--patience', action='store', default=10, type=int,
                        help='set the num of epochs to be patient')
    parser.add_argument('--lr', action='store', default=0.001, type=float,
                        help='set the learning rate of training')
    parser.add_argument('--threads', '-t', action='store', default=4, type=int,
                        help='set the max num of threads')
    parser.add_argument('--seed', '-s', action='store', default=1, type=int,
                        help='set the seed for generating random numbers')
    parser.add_argument('--device', '-d', action='store', default='-1',
                        help='set which device to use')
    parser.add_argument('--file', '-f', action='store', default='model.pt',
                        help='set where to store the model')
    args = parser.parse_args()

    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # read config according to the corresponding model
    config = config.config[args.model]

    print("Preprocess the data")
    train = Corpus(fname=config.ftrain)
    dev = Corpus(fname=config.fdev)
    test = Corpus(fname=config.ftest)
    embed = Embedding(fname=config.fembed)
    vocab = Vocab.from_corpus(corpus=train, min_freq=2)
    vocab.read_embeddings(embed=embed, unk='unk', init_unk=init_embedding)
    print(vocab)

    print("Load the dataset")
    trainset = TextDataset(numericalize(vocab=vocab,
                                        corpus=train,
                                        use_char=config.use_char,
                                        use_elmo=config.use_elmo))
    devset = TextDataset(numericalize(vocab=vocab,
                                      corpus=dev,
                                      use_char=config.use_char,
                                      use_elmo=config.use_elmo))
    testset = TextDataset(numericalize(vocab=vocab,
                                       corpus=test,
                                       use_char=config.use_char,
                                       use_elmo=config.use_elmo))
    print(f"  size of trainset: {len(trainset)}")
    print(f"  size of devset: {len(devset)}")
    print(f"  size of testset: {len(testset)}")

    print("Create Neural Network")
    if args.model == 'char_lstm_crf':
        params = {
            'n_vocab': vocab.n_words,
            'n_embed': config.n_embed,
            'n_char': vocab.n_chars,
            'n_char_embed': config.n_char_embed,
            'n_char_out': config.n_char_out,
            'n_hidden': config.n_hidden,
            'n_out': vocab.n_tags,
            'drop': args.drop
        }
        MODEL = CHAR_LSTM_CRF
    elif args.model == 'elmo_lstm_crf':
        params = {
            'n_vocab': vocab.n_words,
            'n_embed': config.n_embed,
            'n_elmo': config.n_elmo,
            'n_hidden': config.n_hidden,
            'n_out': vocab.n_tags,
            'drop': args.drop
        }
        MODEL = ELMO_LSTM_CRF
    for k, v in params.items():
        print(f"  {k}: {v}")

    model = MODEL(**params)
    model.load_pretrained(vocab.embeddings)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"{model}\n")

    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    dev_loader = DataLoader(dataset=devset,
                            batch_size=args.batch_size,
                            collate_fn=collate_fn)
    test_loader = DataLoader(dataset=testset,
                             batch_size=args.batch_size,
                             collate_fn=collate_fn)

    trainer = Trainer(model=model,
                      vocab=vocab,
                      optimizer=optimizer,
                      task=args.task)
    trainer.fit(train_loader=train_loader,
                dev_loader=dev_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                patience=args.patience,
                file=args.file)
