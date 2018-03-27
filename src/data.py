#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()


class LabeledDataset(Dataset):
    CONSTANTS = {
        'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3, '<STEP>': 4
    }

    def __init__(self, stream_or_file, vocab_enc=None, vocab_dec=None, frozen=False):
        super().__init__()
        self.create_vocabs(vocab_enc, vocab_dec, frozen)
        if isinstance(stream_or_file, str):
            with open(stream_or_file) as stream:
                self._load_stream(stream)
        else:
            self._load_stream(stream)

    def create_vocabs(self, vocab_enc, vocab_dec, frozen):
        self.frozen = frozen
        if vocab_enc is None:
            self.vocab_enc = LabeledDataset.CONSTANTS.copy()
        else:
            self.vocab_enc = vocab_enc
        if vocab_dec is None:
            self.vocab_dec = LabeledDataset.CONSTANTS.copy()
        else:
            self.vocab_dec = vocab_dec

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

        x = []
        y = []

        for i in range(len(self.raw_enc)):
            UNK = self.vocab_enc['UNK']
            enc = self.raw_enc[i]
            padded = enc + ['PAD'] * (self.maxlen_enc-len(enc))
            if self.frozen:
                x.append([self.vocab_enc.get(c, UNK) for c in padded])
            else:
                x.append([self.vocab_enc.setdefault(c, len(self.vocab_enc)) for c in padded])

            dec = self.raw_dec[i]
            padded = dec + ['EOS'] + ['PAD'] * (self.maxlen_dec-len(dec))
            if self.frozen:
                y.append([self.vocab_dec.get(c, UNK) for c in padded])
            else:
                y.append([self.vocab_dec.setdefault(c, len(self.vocab_dec)) for c in padded])

        self.enc_len = np.array([len(enc) for enc in self.raw_enc])
        self.dec_len = np.array([len(dec)+1 for dec in self.raw_dec])
        self.maxlen_dec += 1
        self.X = np.array(x)
        self.Y = np.array(y)

    def save_vocabs(self, expdir):
        enc_fn = os.path.join(expdir, 'vocab_enc')
        with open(enc_fn, 'w') as f:
            f.write("\n".join("{}\t{}".format(k, v) for k, v in sorted(self.vocab_enc.items())))

        dec_fn = os.path.join(expdir, 'vocab_dec')
        with open(dec_fn, 'w') as f:
            f.write("\n".join("{}\t{}".format(k, v) for k, v in sorted(self.vocab_dec.items())))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.enc_len[idx], self.dec_len[idx]
