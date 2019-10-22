# http://anie.me/On-Torchtext/
import revtok

import torch
import torch.nn as nn

from torchtext.vocab import GloVe
from torchtext import data
from torchtext.data import Field
from torchtext.data import Iterator, BucketIterator

class SplitReversibleField(Field):

    def __init__(self, **kwargs):
        if kwargs.get('tokenize') is list:
            self.use_revtok = False
        else:
            self.use_revtok = True
        super(SplitReversibleField, self).__init__(**kwargs)

    def reverse(self, batch, mode='resp'):
        if self.use_revtok:
            try:
                import revtok
            except ImportError:
                print("Please install revtok.")
                raise
        if mode =='resp' and not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        detokenized = []
        for ex in batch:
            if mode=='resp':
                for ind in ex:
                    detokenized.append(self.vocab.itos[ind])
            else:
                detokenized.append(self.vocab.itos[ex])
        batch = detokenized  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        if self.use_revtok:
            return [revtok.detokenize(ex) for ex in batch]
        return [' '.join(ex) for ex in batch]