import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transvoice.models.components.DualPath_Transformer import DualPath

class Separator(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_dim,
        hidden_dim,
        output_dim,
        num_spk=2,
        layer=4,
        dropout=0.1,
        segment_size=100,
        input_normalize=True,
        nheads=4,
    ):
        super(Separator, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_spk = num_spk
        self.layer = layer
        self.dropout = dropout
        self.segment_size = segment_size
        self.input_normalize = input_normalize
        self.nheads = nheads

        self.transformer_model = DualPath(
            self.feature_dim,
            self.hidden_dim,
            self.feature_dim,
            self.num_spk,
            dropout=self.dropout,
            nheads=self.nheads,
            num_layers=self.layer,
            input_normalize=input_normalize,
            segment_size=self.segment_size,
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
        # print(f"[Separator] Input shape: {input.shape}")
        enc_segments, enc_rest = self.create_chuncks(input, self.segment_size)
        # print(
        #     f"[Separator] After create_chuncks, enc_segments shape: {enc_segments.shape}"
        # )
        # separate
        output_all = self.transformer_model(enc_segments)

        # merge back audio files
        output_all_wav = []
        for ii in range(len(output_all)):
            output_ii = self.merge_chuncks(output_all[ii], enc_rest)
            output_all_wav.append(output_ii)
        return output_all_wav
