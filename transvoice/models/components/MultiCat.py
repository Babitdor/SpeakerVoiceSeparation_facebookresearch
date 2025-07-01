import torch
import torch.nn as nn
from transvoice.models.components.InterIntraTransformer import TransformerBlock


class GatedDualTransformerBlock(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0.1,
        nheads=8,
        num_layers=2,
        segment_size=100,
        re_encode_pos=True,
    ):
        super(GatedDualTransformerBlock, self).__init__()

        assert input_size % nheads == 0, "input_size must be divisible by nheads"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.nheads = nheads
        self.num_layers = num_layers
        self.segment_size = segment_size
        self.re_encode_pos = re_encode_pos
        self.transformer = TransformerBlock(
            d_model=self.input_size,
            ff_dims=self.hidden_size,
            dropout=self.dropout,
            num_heads=self.nheads,
            num_layers=self.num_layers,
            positional_encoding_type="relative" if re_encode_pos else "absolute",
            max_len=512,
        )
        self.layer_norm = nn.LayerNorm(self.input_size)
        self.block_projection = nn.Linear(self.input_size * 2, self.input_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.input_size),
        )

    def forward(self, input):

        # print("[MulCat] Input: ", input.shape)
        transformer_out = self.layer_norm(self.transformer(input))
        # apply gated rnn
        gate = torch.sigmoid(
            self.block_projection(torch.cat([transformer_out, input], dim=-1))
        )
        gated = gate * transformer_out + (1 - gate) * input

        # Apply feed-forward network
        output = self.ffn(gated)
        # print("[MulCat] Output: ", output.shape)
        return output
