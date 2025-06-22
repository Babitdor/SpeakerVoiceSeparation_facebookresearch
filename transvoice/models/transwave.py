# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Eliya Nachmani (enk100), Yossi Adi (adiyoss), Lior Wolf

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import overlap_and_add
from ..utils import capture_init
import math


class MulCatBlock(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0, nheads=4, num_layers=4):
        super(MulCatBlock, self).__init__()

        assert input_size % nheads == 0, "input_size must be divisible by nheads"

        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.pos_encode = nn.Parameter(torch.randn(1, 1000, input_size) * 0.01)
        self.pos_encode = SinusoidalPositionalEncoding(input_size)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=nheads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(input_size)
        self.block_projection = nn.Linear(input_size * 2, input_size)

    def forward(self, input):
        # input (batch, seq, feature)
        # print(input.shape)
        # seq_len = input.shape[1]
        pos_enc = self.pos_encode(input)
        transformer_in = input + pos_enc

        transformer_out = self.layer_norm(self.transformer(transformer_in))
        # apply gated rnn
        gated_output = torch.mul(transformer_out, input)
        gated_output = torch.cat([gated_output, input], dim=2)
        gated_output = self.block_projection(
            gated_output.contiguous().view(-1, gated_output.shape[2])
        ).view(input.shape)
        return gated_output


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq, feature)
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class ByPass(nn.Module):
    def __init__(self):
        super(ByPass, self).__init__()

    def forward(self, input):
        return input


class DPMulCat(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_spk,
        dropout=0.1,
        nheads=4,
        num_layers=1,
        input_normalize=False,
    ):
        super(DPMulCat, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.in_norm = input_normalize
        self.num_layers = num_layers

        self.rows_grnn = nn.ModuleList([])
        self.cols_grnn = nn.ModuleList([])
        self.rows_normalization = nn.ModuleList([])
        self.cols_normalization = nn.ModuleList([])

        # create the dual path pipeline
        for i in range(num_layers):
            self.rows_grnn.append(MulCatBlock(input_size, hidden_size, dropout, nheads))
            self.cols_grnn.append(MulCatBlock(input_size, hidden_size, dropout, nheads))
            if self.in_norm:
                self.rows_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
                self.cols_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
            else:
                # used to disable normalization
                self.rows_normalization.append(ByPass())
                self.cols_normalization.append(ByPass())

        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(input_size, output_size * num_spk, 1)
        )

    def forward(self, input):
        batch_size, _, d1, d2 = input.shape
        output = input
        output_all = []
        for i in range(self.num_layers):
            row_input = output.permute(0, 3, 2, 1).reshape(batch_size * d2, d1, -1)
            row_output = self.rows_grnn[i](row_input)
            row_output = row_output.view(batch_size, d2, d1, -1).permute(0, 3, 2, 1)
            row_output = self.rows_normalization[i](row_output)
            # apply a skip connection
            if self.training:
                output = output + row_output
            else:
                output += row_output

            col_input = output.permute(0, 2, 3, 1).reshape(batch_size * d1, d2, -1)
            col_output = self.cols_grnn[i](col_input)
            col_output = col_output.view(batch_size, d1, d2, -1).permute(0, 3, 1, 2)
            output = self.cols_normalization[i](col_output).contiguous()
            # apply a skip connection
            if self.training:
                output = output + col_output
            else:
                output += col_output

            output_i = self.output(output)
            if self.training or i == (self.num_layers - 1):
                output_all.append(output_i)
        return output_all


class Separator(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim,
        output_dim,
        num_spk=2,
        layer=4,
        segment_size=100,
        input_normalize=True,
        nheads=4,
    ):
        super(Separator, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer = layer
        self.segment_size = segment_size
        self.nheads = nheads
        self.num_spk = num_spk
        self.input_normalize = input_normalize

        self.rnn_model = DPMulCat(
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim,
            self.num_spk,
            dropout=0.1,
            nheads=self.nheads,
            num_layers=layer,
            input_normalize=input_normalize,
        )

    # ======================================= #
    # The following code block was borrowed and modified from https://github.com/yluo42/TAC
    # ================ BEGIN ================ #
    def pad_segment(self, input, segment_size):
        # input is the features: (B, N, T)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2
        rest = segment_size - (segment_stride + seq_len % segment_size) % segment_size
        if rest > 0:
            pad = torch.zeros(
                batch_size, dim, rest, device=input.device, dtype=input.dtype
            )
            input = torch.cat([input, pad], 2)

        pad_aux = torch.zeros(
            batch_size, dim, segment_stride, device=input.device, dtype=input.dtype
        )
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    def create_chuncks(self, input, segment_size):
        # split the feature into chunks of segment size
        # input is the features: (B, N, T)

        input, rest = self.pad_segment(input, segment_size)
        batch_size, dim, seq_len = input.shape
        segment_stride = segment_size // 2

        segments1 = (
            input[:, :, :-segment_stride]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments2 = (
            input[:, :, segment_stride:]
            .contiguous()
            .view(batch_size, dim, -1, segment_size)
        )
        segments = (
            torch.cat([segments1, segments2], 3)
            .view(batch_size, dim, -1, segment_size)
            .transpose(2, 3)
        )
        return segments.contiguous(), rest

    def merge_chuncks(self, input, rest):
        # merge the splitted features into full utterance
        # input is the features: (B, N, L, K)

        batch_size, dim, segment_size, _ = input.shape
        segment_stride = segment_size // 2
        input = (
            input.transpose(2, 3)
            .contiguous()
            .view(batch_size, dim, -1, segment_size * 2)
        )  # B, N, K, L

        input1 = (
            input[:, :, :, :segment_size]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, segment_stride:]
        )
        input2 = (
            input[:, :, :, segment_size:]
            .contiguous()
            .view(batch_size, dim, -1)[:, :, :-segment_stride]
        )

        output = input1 + input2
        if rest > 0:
            output = output[:, :, :-rest]
        return output.contiguous()  # B, N, T

    # ================= END ================= #

    def forward(self, input):
        # create chunks
        enc_segments, enc_rest = self.create_chuncks(input, self.segment_size)
        # separate
        output_all = self.rnn_model(enc_segments)

        # merge back audio files
        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_chuncks(output_all[ii], enc_rest)
            output_all_wav.append(output_ii)
        return output_all_wav


class SWave(nn.Module):
    @capture_init
    def __init__(self, N, L, H, R, C, sr, segment, input_normalize, nheads=4):
        super(SWave, self).__init__()
        # hyper-parameter
        self.N, self.L, self.H, self.R, self.C, self.sr, self.segment = (
            N,
            L,
            H,
            R,
            C,
            sr,
            segment,
        )
        self.nheads = nheads
        self.input_normalize = input_normalize
        self.context_len = 2 * self.sr / 1000
        self.context = int(self.sr * self.context_len / 1000)
        self.layer = self.R
        self.filter_dim = self.context * 2 + 1
        self.num_spk = self.C
        # similar to dprnn paper, setting chancksize to sqrt(2*L)
        self.segment_size = int(np.sqrt(2 * self.sr * self.segment / (self.L / 2)))

        # model sub-networks
        self.encoder = Encoder(L, N)
        self.decoder = Decoder(L)
        self.separator = Separator(
            self.filter_dim + self.N,
            self.N,
            self.H,
            self.filter_dim,
            self.num_spk,
            self.layer,
            self.segment_size,
            self.input_normalize,
            self.nheads,
        )
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        # print(mixture.shape)
        mixture_w = self.encoder(mixture)
        output_all = self.separator(mixture_w)

        # fix time dimension, might change due to convolution operations
        T_mix = mixture.size(-1)
        # generate wav after each RNN block and optimize the loss
        outputs = []
        for ii in range(len(output_all)):
            output_ii = output_all[ii].view(
                mixture.shape[0], self.C, self.N, mixture_w.shape[2]
            )
            # print(
            #     "Before decoder stats:",
            #     output_ii.min().item(),
            #     output_ii.max().item(),
            #     output_ii.mean().item(),
            # )
            output_ii = self.decoder(output_ii)

            T_est = output_ii.size(-1)
            output_ii = F.pad(output_ii, (0, T_mix - T_est))
            outputs.append(output_ii)

            # print(outputs.__sizeof__)
        return torch.stack(outputs)


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        # setting 50% overlap
        self.conv = nn.Sequential(
            nn.Conv1d(1, N, kernel_size=L, stride=L // 2, groups=1, bias=False),
            nn.Conv1d(N, N, kernel_size=1, bias=False),
        )

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = F.relu(self.conv(mixture))
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, L):
        super(Decoder, self).__init__()
        self.L = L
        self.GLU = torch.nn.GLU()

    def forward(self, est_source):
        est_source = torch.transpose(est_source, 2, 3)
        est_source = F.avg_pool2d(est_source, (1, self.L))
        est_source = overlap_and_add(est_source, self.L // 2)
        # est_source = self.GLU(est_source)
        return est_source
