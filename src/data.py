#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()


class LabeledDataset(Dataset):
    def __init__(self, stream):
        super().__init__()
        self._load_stream(stream)

    def _load_stream(self, stream):
        self.raw_enc = []
        self.raw_dec = []
        for line in stream:
            enc, dec = line.rstrip("\n").split("\t")
            self.raw_enc.append(enc.split(" "))
            self.raw_dec.append(dec.split(" "))

        self.maxlen_enc = max(len(r) for r in self.raw_enc)
        self.maxlen_dec = max(len(r) for r in self.raw_dec)

        self._create_padded_matrices()

    def _create_padded_matrices(self):
        self.vocab_enc = {'PAD': 0, 'UNK': 3}
        self.vocab_dec = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3, '<STEP>': 4}

        x = []
        y = []

        for i in range(len(self.raw_enc)):
            UNK = self.vocab_enc['UNK']
            enc = self.raw_enc[i]
            padded = enc + ['PAD'] * (self.maxlen_enc-len(enc))
            x.append([self.vocab_enc.setdefault(c, len(self.vocab_enc)) for c in padded])

            dec = self.raw_dec[i]
            padded = dec + ['EOS'] + ['PAD'] * (self.maxlen_dec-len(dec))
            y.append([self.vocab_dec.setdefault(c, len(self.vocab_dec)) for c in padded])

        self.enc_len = [len(enc) for enc in self.raw_enc]
        self.dec_len = [len(dec)+1 for dec in self.raw_dec]
        self.maxlen_dec += 1
        self.X = np.array(x)
        self.Y = np.array(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.enc_len[idx], self.dec_len[idx]

    @staticmethod
    def create_sos_vector(batch_size):
        vec = Variable(torch.LongTensor(np.ones(batch_size)))
        if use_cuda:
            vec = vec.cuda()
        return vec
