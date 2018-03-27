#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
from sys import stdin

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from loss import masked_cross_entropy
from data import LabeledDataset


def parse_args():
    p = ArgumentParser()
    return p.parse_args()


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.cell = nn.LSTM(embedding_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        nn.init.xavier_uniform(self.embedding.weight)

        self.hidden_size = hidden_size

    def forward(self, input):
        embedded = self.embedding(input)
        outputs, hidden = self.cell(embedded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, embedding_size)
        nn.init.xavier_uniform(self.embedding.weight)
        self.cell = nn.LSTM(embedding_size + hidden_size, hidden_size,
                            num_layers=num_layers, bidirectional=False)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, encoder_output, last_hidden):
        bs = input_seq.size(0)
        embedded = self.embedding(input_seq)
        rnn_input = torch.cat((embedded, encoder_output), 1)
        rnn_input = rnn_input.view(1, *rnn_input.size())
        rnn_output, hidden = self.cell(rnn_input, last_hidden)
        output = self.out(rnn_output)
        return output, hidden


def train(encoder, decoder, data):
    encoder.train(True)
    decoder.train(True)
    enc_opt = optim.Adam(encoder.parameters())
    dec_opt = optim.Adam(decoder.parameters())
    for epoch in range(200):
        epoch_loss = 0
        for i, batch in enumerate(data):
            X = Variable(batch[0])
            Y = Variable(batch[1])
            tgt_len = Variable(batch[3], requires_grad=False)
            if use_cuda:
                X = X.cuda()
                Y = Y.cuda()
                tgt_len = tgt_len.cuda()
            outputs, enc_hidden = encoder(X)
            all_output = Variable(torch.zeros(Y.size(0), Y.size(1), len(data.dataset.vocab_dec)))
            if use_cuda:
                all_output = all_output.cuda()

            enc_out = outputs
            dec_input = LabeledDataset.create_sos_vector(X.size(0))
            attn_pos = Variable(torch.LongTensor([0] * X.size(0)))
            range_helper = Variable(torch.LongTensor(np.arange(X.size(0))), requires_grad=False)
            if use_cuda:
                attn_pos = attn_pos.cuda()
                range_helper = range_helper.cuda()
            hidden = tuple(e[[0], :, :].contiguous() for e in enc_hidden)
            for ti in range(Y.size(1)):
                out, hidden = decoder(dec_input, enc_out[range_helper, attn_pos], hidden)
                topv, topi = out.max(-1)
                attn_pos = attn_pos + torch.eq(topi, 4).long()
                attn_pos = torch.clamp(attn_pos, 0, X.size(1)-1)
                attn_pos = attn_pos.squeeze(0).contiguous()
                dec_input = Y[:, ti].contiguous()
                all_output[:, ti] = out

            enc_opt.zero_grad()
            dec_opt.zero_grad()
            loss = masked_cross_entropy(all_output.contiguous(), Y, tgt_len)
            epoch_loss += loss.data[0]
            loss.backward()
            enc_opt.step()
            dec_opt.step()
        print(epoch, epoch_loss / (i+1))

    encoder.train(False)
    decoder.train(False)

def main():

    data = LabeledDataset(stdin)
    loader = DataLoader(data, batch_size=128)

    hidden_size = 128
    embedding_size = 30
    encoder = EncoderRNN(len(data.vocab_enc), embedding_size, hidden_size, 1)
    decoder = DecoderRNN(len(data.vocab_dec), embedding_size, hidden_size ,1)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    train(encoder, decoder, loader)

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    main()
