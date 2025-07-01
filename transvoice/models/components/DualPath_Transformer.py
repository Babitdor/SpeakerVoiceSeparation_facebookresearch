import torch.nn as nn
import torch.nn.functional as F
from transvoice.models.components.MultiCat import GatedDualTransformerBlock
import torch


class DualPath(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_spk,
        dropout=0.1,
        nheads=4,
        num_layers=6,
        input_normalize=False,
        segment_size=100,
    ):
        super(DualPath, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_spk = num_spk
        self.dropout = dropout
        self.nheads = nheads
        self.num_layers = num_layers
        self.in_norm = input_normalize
        self.segment_size = segment_size

        self.IntraTransformer = nn.ModuleList([])
        self.InterTransformer = nn.ModuleList([])
        self.Intra_normalization = nn.ModuleList([])
        self.Inter_normalization = nn.ModuleList([])
        self.res_scale = nn.Parameter(torch.ones(1))

        # create the dual path pipeline
        for i in range(num_layers):
            self.IntraTransformer.append(
                GatedDualTransformerBlock(
                    self.input_size,
                    self.hidden_size,
                    self.dropout,
                    self.nheads,
                    self.num_layers,  # Fixed: use smaller num_layers for individual blocks
                    self.segment_size,
                    re_encode_pos=True,
                )
            )
            self.InterTransformer.append(
                GatedDualTransformerBlock(
                    self.input_size,
                    self.hidden_size,
                    self.dropout,
                    self.nheads,
                    self.num_layers,  # Fixed: use smaller num_layers for individual blocks
                    self.segment_size,
                    re_encode_pos=True,
                )
            )
            if self.in_norm:
                self.Intra_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
                self.Inter_normalization.append(nn.GroupNorm(1, input_size, eps=1e-8))
            else:
                # used to disable normalization
                self.Intra_normalization.append(ByPass())
                self.Inter_normalization.append(ByPass())

        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(input_size, output_size * num_spk, 1)
        )

    def forward(self, input):
        batch_size, _, d1, d2 = input.shape
        output = input
        output_all = []

        for i in range(self.num_layers):
            # Process rows (intra-chunk processing)
            intra_input = output.permute(0, 3, 2, 1).reshape(batch_size * d2, d1, -1)
            intra_output = self.IntraTransformer[i](intra_input)
            intra_output = intra_output.view(batch_size, d2, d1, -1).permute(0, 3, 2, 1)
            intra_output = self.Intra_normalization[i](intra_output)

            # Apply skip connection
            output = output + self.res_scale * intra_output

            # Process columns (inter-chunk processing)
            inter_input = output.permute(0, 2, 3, 1).reshape(batch_size * d1, d2, -1)
            inter_output = self.InterTransformer[i](inter_input)
            inter_output = inter_output.view(batch_size, d1, d2, -1).permute(0, 3, 1, 2)
            inter_output = self.Inter_normalization[i](inter_output)

            # Apply skip connection
            output = output + self.res_scale * inter_output

            # Generate output for this layer
            output_i = self.output(output)
            if self.training or i == (self.num_layers - 1):
                output_all.append(output_i)

        return output_all


class ByPass(nn.Module):
    def __init__(self):
        super(ByPass, self).__init__()

    def forward(self, input):
        return input
