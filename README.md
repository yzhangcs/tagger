# Tagger

Models used for Sequence Labeling

## Requirements

```txt
allennlp == 0.7.0
python == 3.7.0
pytorch == 0.4.1
```

## Usage

### Commands

```sh
$ git clone https://github.com/zysite/tagger.git
$ cd tagger
# eg: BiLSTM+CHAR+CRF
$ python run.py --model=char_lstm_crf --task=ner
```

### Arguments

```sh
$ python run.py -h
usage: run.py [-h] [--model {char_lstm_crf,elmo_lstm_crf}]
              [--task {chunking,ner,pos}] [--drop DROP]
              [--batch_size BATCH_SIZE] [--epochs EPOCHS]
              [--patience PATIENCE] [--lr LR] [--threads THREADS]
              [--seed SEED] [--device DEVICE] [--file FILE]

Create several models for Sequence Labeling.

optional arguments:
  -h, --help            show this help message and exit
  --model {char_lstm_crf,elmo_lstm_crf}
                        choose the model for Sequence Labeling
  --task {chunking,ner,pos}
                        choose the task of Sequence Labeling
  --drop DROP           set the prob of dropout
  --batch_size BATCH_SIZE
                        set the size of batch
  --epochs EPOCHS       set the max num of epochs
  --patience PATIENCE   set the num of epochs to be patient
  --lr LR               set the learning rate of training
  --threads THREADS, -t THREADS
                        set the max num of threads
  --seed SEED, -s SEED  set the seed for generating random numbers
  --device DEVICE, -d DEVICE
                        set which device to use
  --file FILE, -f FILE  set where to store the model
```

## Structures

```python
# CHAR+BiLSTM+CRF
CHAR_LSTM_CRF(
  (embed): Embedding(405440, 100)
  (char_lstm): CharLSTM(
    (embed): Embedding(517, 30)
    (lstm): LSTM(30, 150, batch_first=True, bidirectional=True)
  )
  (word_lstm): LSTM(400, 150, batch_first=True, bidirectional=True)
  (hid): Sequential(
    (0): Linear(in_features=300, out_features=150, bias=True)
    (1): Tanh()
  )
  (out): Linear(in_features=150, out_features=17, bias=True)
  (crf): CRF()
  (drop): Dropout(p=0.5)
)
# ELMo+BiLSTM+CRF
ELMO_LSTM_CRF(
  (embed): Embedding(405440, 100)
  (scalar_mix): ScalarMix()
  (lstm): LSTM(1124, 150, batch_first=True, bidirectional=True)
  (hid): Sequential(
    (0): Linear(in_features=300, out_features=150, bias=True)
    (1): Tanh()
  )
  (out): Linear(in_features=150, out_features=17, bias=True)
  (crf): CRF()
  (drop): Dropout(p=0.5)
)
```

## Results

### NER

* Pretrained: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/).
* Dataset: CoNLL-2003

|               | Dev    | Test   | mT(s)          |
| :-----------: | :----: | :----: | :------------: |
| CHAR_LSTM_CRF | 94.49% | 90.72% | 0:01:50.889580 |
| ELMO_LSTM_CRF | 95.64% | 92.09% | 0:01:46.960411 |

### Chunking

* Pretrained: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/).
* Dataset: CoNLL-2000

|               | Dev    | Test   | mT(s)          |
| :-----------: | :----: | :----: | :------------: |
| CHAR_LSTM_CRF | 95.02% | 94.51% | 0:01:21.141716 |
| ELMO_LSTM_CRF | 97.08% | 96.34% | 0:01:14.761098 |

### POS-Tagging

* Pretrained: [glove.6B.100d.txt](https://nlp.stanford.edu/projects/glove/).
* Dataset: WSJ

|               | Dev    | Test   | mT(s)          |
| :-----------: | :----: | :----: | :------------: |
| CHAR_LSTM_CRF | 97.68% | 97.64% | 0:05:59.462637 |
| ELMO_LSTM_CRF | 97.86% | 97.81% | 0:05:55.335100 |

## References

* [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)
* [Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
* [Empower Sequence Labeling with Task-Aware Neural Language Model](https://arxiv.org/abs/1709.04109)