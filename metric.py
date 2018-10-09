# -*- coding: utf-8 -*-


class AccuracyMethod(object):

    def __init__(self, eps=1e-5):
        super(AccuracyMethod, self).__init__()

        self.tp = 0.0
        self.total = 0.0
        self.eps = eps

    def __call__(self, predicts, targets):
        for predict, target in zip(predicts, targets):
            self.tp += torch.sum(predict == target).item()
            self.total += len(target)

    @property
    def accuracy(self):
        return self.tp / (self.total + self.eps)


class SpanF1Method(object):

    def __init__(self, tags, eps=1e-5):
        super(SpanF1Method, self).__init__()

        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.tags = tags
        self.eps = eps

    def __call__(self, predicts, targets):
        for predict, target in zip(predicts, targets):
            pred_spans = self.get_entities(predict)
            gold_spans = self.get_entities(target)
            self.tp += len(pred_spans & gold_spans)
            self.pred += len(pred_spans)
            self.gold += len(gold_spans)

    @property
    def precision(self):
        return self.tp / (self.pred + self.eps)

    @property
    def recall(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f_score(self):
        precision = self.precision
        recall = self.recall

        return 2 * precision * recall / (precision + recall + self.eps)

    def get_entities(self, tiseq):
        span, chunks = [], []
        tagseq = (self.tags[ti] for ti in tiseq)

        for i, tag in enumerate(tagseq):
            if tag == 'O':
                stype = 'O'
            else:
                stype, etype = tag.split('-')
            if stype == 'B':
                if span:
                    chunks.append(tuple(span))
                span = [etype, i, 1]
            elif stype == 'S':
                if span:
                    chunks.append(tuple(span))
                    span.clear()
                chunks.append((etype, i, 1))
            elif stype == 'I':
                if span:
                    if etype == span[0]:
                        span[-1] += 1
                    else:
                        chunks.append(tuple(span))
                        span = [etype, i, 1]
                else:
                    span = [etype, i, 1]
            elif stype == 'E':
                if span:
                    if etype == span[0]:
                        span[-1] += 1
                        chunks.append(tuple(span))
                        span.clear()
                    else:
                        chunks.append(tuple(span))
                        span.clear()
                        chunks.append((etype, i, 1))
                else:
                    chunks.append((etype, i, 1))
                    span.clear()
            else:
                if span:
                    chunks.append(tuple(span))
                span.clear()
        if span:
            chunks.append(tuple(span))

        return set(chunks)