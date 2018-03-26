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
from torch.utils.data import Dataset, DataLoader

from loss import masked_cross_entropy


def parse_args():
    p = ArgumentParser()
    return p.parse_args()


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
        # embedded = embedded.view(1, batch_size, embedded.size(-1))
        rnn_input = torch.cat((embedded, encoder_output.unsqueeze(0)), 1)
        rnn_input = rnn_input.view(1, *rnn_input.size())
        rnn_output, hidden = self.cell(rnn_input, last_hidden)
        # out = F.softmax(rnn_output, dim=-1)
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
            for bi in range(Y.size(0)):
                y = Y[bi]
                enc_out = outputs[bi]
                dec_input = LabeledDataset.create_sos_vector(1)
                attn_pos = 0
                hidden = tuple(e[[0], [bi], :].unsqueeze(1).contiguous() for e in enc_hidden)
                all_attn_pos = []
                for ti in range(Y.size(1)):
                    out, hidden = decoder(dec_input, enc_out[attn_pos], hidden)
                    topv, topi = out.data.topk(1)
                    # FIXME hardcoded STEP symbol (4)
                    if int(topi[0][0]) == 4:
                        attn_pos = min(attn_pos+1, enc_out.size(0)-1)
                    all_attn_pos.append(attn_pos)
                    dec_input = Y[bi, ti]
                    all_output[bi, ti] = out
                if epoch % 10 == 9 and bi < 5:
                    print(" ".join(map(str, all_attn_pos)))
            enc_opt.zero_grad()
            dec_opt.zero_grad()
            loss = masked_cross_entropy(all_output.contiguous(), Y, tgt_len)
            epoch_loss += loss.data[0]
            loss.backward()
            enc_opt.step()
            dec_opt.step()
        print(epoch, epoch_loss)

    encoder.train(False)
    decoder.train(False)

def main():
    # args = parse_args()

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
