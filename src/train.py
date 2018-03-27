#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin
from datetime import datetime
import yaml
import os
import logging

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss import masked_cross_entropy
from data import LabeledDataset
from model import MonotonicAttentionSeq2seq


def parse_args():
    p = ArgumentParser()
    p.add_argument('-t', '--train-file', type=str, required=True)
    p.add_argument('-d', '--dev-file', type=str, required=True)
    p.add_argument('-c', '--config', type=str, required=True)
    p.add_argument('--nosave', action='store_true',
                   help="Do not save experiment results")
    return p.parse_args()


class Config:
    @classmethod
    def from_yaml(cls, filename):
        c = cls()
        with open(filename) as f:
            c.params = yaml.load(f)
            for k, v in c.params.items():
                setattr(c, k, v)
        return c

    def generate_expdir(self):
        i = 0
        fmt = '{0:04d}'
        while os.path.exists(os.path.join(self.experiment_dir,
                                            fmt.format(i))):
            i += 1
        self.experiment_dir = os.path.join(
            self.experiment_dir, fmt.format(i))
        os.makedirs(self.experiment_dir)
        return self.experiment_dir

    def save(self):
        d = self.params
        d['experiment_dir'] = self.experiment_dir
        with open(os.path.join(self.experiment_dir, 'config.yaml'), 'w') as f:
            yaml.dump(d, f, default_flow_style=False)


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)

    train_data = LabeledDataset(args.train_file)
    train_loader = DataLoader(train_data, batch_size=config.batch_size)

    dev_data = LabeledDataset(args.dev_file, vocab_enc=train_data.vocab_enc,
                              vocab_dec=train_data.vocab_dec, frozen=True)
    dev_loader = DataLoader(dev_data, batch_size=config.batch_size)


    model = MonotonicAttentionSeq2seq(config, len(train_data.vocab_enc),
                                      len(train_data.vocab_dec))
    if use_cuda:
        model.cuda()

    config.generate_expdir()
    config.save()
    train_data.save_vocabs(config.experiment_dir)

    model.train(train_loader, dev_loader, nosave=args.nosave)
    model.result.save(config.experiment_dir)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    main()
