#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os
import logging
import yaml
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

from loss import masked_cross_entropy
from data import LabeledDataset

use_cuda = torch.cuda.is_available()


class EncoderRNN(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config
        self.embedding_size = config.embedding_size_enc
        self.hidden_size = config.hidden_size_enc
        self.num_layers = config.num_layers_enc
        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.cell = nn.LSTM(self.embedding_size, self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)
        nn.init.xavier_uniform(self.embedding.weight)

    def forward(self, input):
        embedded = self.embedding(input)
        outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size_dec
        self.output_size = output_size
        self.embedding_size = config.embedding_size_dec
        self.num_layers = config.num_layers_dec

        self.embedding = nn.Embedding(output_size, self.embedding_size)
        nn.init.xavier_uniform(self.embedding.weight)
        self.cell = nn.LSTM(self.embedding_size + self.hidden_size, self.hidden_size,
                            num_layers=self.num_layers, bidirectional=False)
        self.output_proj = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_seq, encoder_output, last_hidden):
        embedded = self.embedding(input_seq)
        rnn_input = torch.cat((embedded, encoder_output), 1)
        rnn_input = rnn_input.view(1, *rnn_input.size())
        rnn_output, hidden = self.cell(rnn_input, last_hidden)
        output = self.output_proj(rnn_output)
        return output, hidden


class MonotonicAttentionSeq2seq(nn.Module):
    def __init__(self, config, vocab_size_enc, vocab_size_dec):
        super().__init__()
        self.config = config
        self.encoder = EncoderRNN(config, vocab_size_enc)
        self.decoder = DecoderRNN(config, vocab_size_dec)
        self.result = Result()

    def train(self, train_data, dev_data=None, nosave=False):

        self.enc_opt = optim.Adam(self.encoder.parameters())
        self.dec_opt = optim.Adam(self.decoder.parameters())

        self.result.start()
        for epoch in range(self.config.epochs):
            train_loss = self.run_epoch(train_data, do_train=True)
            self.result.train_loss.append(train_loss)
            if dev_data is not None:
                dev_loss = self.run_epoch(dev_data, do_train=False)
                self.result.dev_loss.append(dev_loss)
            else:
                dev_loss = None
            if nosave is False:
                self.save_if_best(train_loss, dev_loss, epoch)
            print(epoch, train_loss, dev_loss)
        self.result.end()

    def save_if_best(self, train_loss, dev_loss, epoch):
        if epoch < self.config.save_min_epoch:
            return
        loss = dev_loss if dev_loss is not None else train_loss
        if not hasattr(self, 'min_loss') or self.min_loss > loss:
            self.min_loss = loss
            save_path = os.path.join(
                self.config.experiment_dir,
                "model.epoch_{}".format("{0:04d}".format(epoch)))
            logging.info("Saving model to {}".format(save_path))
            torch.save(self.state_dict(), save_path)

    def run_epoch(self, data, do_train):
        self.encoder.train(do_train)
        self.decoder.train(do_train)
        epoch_loss = 0
        for bi, batch in enumerate(data):
            X, Y, x_len, y_len = [Variable(b) for b in batch]
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()
                x_len = x_len.cuda()
                y_len = y_len.cuda()
            batch_size = X.size(0)
            seqlen_enc = X.size(1)
            seqlen_dec = Y.size(1)

            enc_outputs, enc_hidden = self.encoder(X)
            all_output = Variable(
                torch.zeros(batch_size, seqlen_dec, len(data.dataset.vocab_dec)))
            dec_input = Variable(torch.LongTensor(
                np.ones(batch_size) * LabeledDataset.CONSTANTS['SOS']))
            attn_pos = Variable(torch.LongTensor([0] * batch_size))
            range_helper = Variable(torch.LongTensor(np.arange(batch_size)),
                                    requires_grad=False)

            if use_cuda:
                all_output = all_output.cuda()
                dec_input = dec_input.cuda()
                attn_pos = attn_pos.cuda()
                range_helper = range_helper.cuda()

            hidden = tuple(e[:self.decoder.num_layers, :, :].contiguous()
                           for e in enc_hidden)

            for ts in range(seqlen_dec):
                dec_out, hidden = self.decoder(
                    dec_input, enc_outputs[range_helper, attn_pos], hidden)
                topv, top_idx = dec_out.max(-1)
                attn_pos = attn_pos + torch.eq(top_idx,
                                               LabeledDataset.CONSTANTS['<STEP>']).long()
                attn_pos = torch.clamp(attn_pos, 0, seqlen_enc-1)
                attn_pos = attn_pos.squeeze(0).contiguous()
                dec_input = Y[:, ts].contiguous()
                all_output[:, ts] = dec_out

            self.enc_opt.zero_grad()
            self.dec_opt.zero_grad()
            loss = masked_cross_entropy(all_output.contiguous(), Y, y_len)
            epoch_loss += loss.data[0]
            if do_train:
                loss.backward()
                self.enc_opt.step()
                self.dec_opt.step()
        epoch_loss /= (bi+1)
        return epoch_loss


class Result:
    __slots__ = ('train_loss', 'dev_loss', 'running_time', 'start_time')
    def __init__(self):
        self.train_loss = []
        self.dev_loss = []

    def start(self):
        self.start_time = datetime.now()

    def end(self):
        self.running_time = (datetime.now() - self.start_time).total_seconds()

    def save(self, expdir):
        d = {k: getattr(self, k) for k in self.__slots__}
        with open(os.path.join(expdir, 'result.yaml'), 'w') as f:
            yaml.dump(d, f, default_flow_style=False)
