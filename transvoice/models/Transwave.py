# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Authors: Eliya Nachmani (enk100), Yossi Adi (adiyoss), Lior Wolf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transvoice.scripts.utils import capture_init
from transvoice.models.components.AudioEncoder import Encoder
from transvoice.models.components.AudioDecoder import Decoder
from transvoice.models.components.Separator import Separator


class TranSWave(nn.Module):
    @capture_init
    def __init__(
        self, N, L, H, R, C, sr, segment, input_normalize, nheads=4, dropout=0.1
    ):
        super(TranSWave, self).__init__()
        # hyper-parameter
        self.N, self.L, self.H, self.R, self.C, self.sr, self.segment = (
            N,  # Output size of the latent Representation
            L,  # Encoding Compression
            H,  # Hidden Size
            R,  # Number of repeated separation blocks.
            C,  # no. of Speakers
            sr,  # Sampling Rate
            segment,  # Length of audio chunks fed into the model
        )
        self.nheads = nheads
        self.input_normalize = input_normalize
        self.context_len = 2 * self.sr / 1000
        self.context = int(self.sr * self.context_len / 1000)
        self.layer = self.R
        self.filter_dim = self.context * 2 + 1
        self.num_spk = self.C
        self.dropout = dropout
        # similar to dprnn paper, setting chancksize to sqrt(2*L)
        self.segment_size = int(np.sqrt(2 * self.sr * self.segment / (self.L / 2)))

        # model sub-networks
        self.encoder = Encoder(L, N)
        self.decoder = Decoder(L)
        self.separator = Separator(
            self.filter_dim + self.N,  # Input Size
            self.N,  # Feature Size
            self.H,  # Hidden Size
            self.filter_dim,  # Output Size
            self.num_spk,  # Number of Speakers
            self.layer,  # Number of Separation Layers
            self.dropout,
            self.segment_size,  # Length of audio chunks fed into the model
            self.input_normalize,  # To normalise Input values
            self.nheads,  # Number of Attention Heads
        )
        # init Parameters using Xavier Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        # print(mixture.shape)
        # print(f"[SWave] Input mixture shape: {mixture.shape}")
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
            output_ii = self.decoder(output_ii)

            T_est = output_ii.size(-1)
            output_ii = F.pad(output_ii, (0, T_mix - T_est))
            outputs.append(output_ii)

            # print(outputs.__sizeof__)
        # print(f"[SWave] Final output shape: {outputs[-1].shape}")
        return torch.stack(outputs)
