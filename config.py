# -*- coding: utf-8 -*-


class Config(object):
    ftrain = 'data/ner/train.txt'
    fdev = 'data/ner/dev.txt'
    ftest = 'data/ner/test.txt'
    fembed = 'data/glove.6B.100d.txt'


class CHAR_LSTM_CRF_Config(Config):
    n_embed = 100
    n_char_embed = 30
    n_char_out = 300
    n_hidden = 150
    use_char = True
    use_elmo = False
    use_parser = False


class ELMO_LSTM_CRF_Config(Config):
    n_embed = 100
    n_elmo = 1024
    n_hidden = 150
    use_char = False
    use_elmo = True
    use_parser = False


class PARSER_LSTM_CRF_Config(Config):
    n_embed = 100
    n_char_embed = 30
    n_char_out = 300
    n_parser = 800
    n_hidden = 150
    use_char = True
    use_elmo = False
    use_parser = True


config = {
    'char_lstm_crf': CHAR_LSTM_CRF_Config,
    'elmo_lstm_crf': ELMO_LSTM_CRF_Config,
    'parser_lstm_crf': PARSER_LSTM_CRF_Config,
}
