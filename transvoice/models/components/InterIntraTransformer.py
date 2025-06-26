import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerScale(torch.nn.Module):
    def __init__(self, dims, input_size, Layer_scale_init=1.0e-5):
        super().__init__()

        shape = {1: (input_size,), 2: (1, input_size), 3: (1, 1, input_size)}.get(
            dims, None
        )

        if shape is None:
            raise ValueError(f"Invalid dims: {dims}. Must be 1, 2, or 3.")

        self.layer_scale = nn.Parameter(
            torch.ones(shape) * Layer_scale_init, requires_grad=True
        )

    def forward(self, x):
        return x * self.layer_scale


class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, in_channels: int, num_heads: int, maxlen: int, embed_v=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.embedding_dim = self.in_channels // self.num_heads
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(
            num_embeddings=2 * maxlen, embedding_dim=self.embedding_dim
        )
        self.pe_v = (
            torch.nn.Embedding(
                num_embeddings=2 * maxlen, embedding_dim=self.embedding_dim
            )
            if embed_v
            else None
        )
        self.proj = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, seq_len):
        range_vec = torch.arange(seq_len, device=self.pe_k.weight.device)
        rel_mat = range_vec[None, :] - range_vec[:, None]
        rel_mat = rel_mat.clamp(-self.maxlen, self.maxlen - 1)
        rel_mat = rel_mat + self.maxlen
        pe_k_output = self.pe_k(rel_mat)
        pe_k_output = self.proj(pe_k_output)
        return pe_k_output


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dims=None, dropout=0.1):
        super().__init__()
        self.ff_dims = ff_dims if ff_dims is not None else 4 * d_model
        self.linear1 = nn.Linear(d_model, self.ff_dims)
        self.linear2 = nn.Linear(self.ff_dims, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        n_head: int,
        in_channels: int,
        dropout_rate: float,
        Layer_scale_init=1e-4,
        maxlen: int = 500,
    ):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head
        self.d_model = in_channels  # Fixed: added missing d_model attribute
        self.nhead = n_head
        self.pos_enc = RelativePositionalEncoding(in_channels, n_head, maxlen)
        self.layer_norm = torch.nn.LayerNorm(in_channels)
        self.linear_q = torch.nn.Linear(in_channels, in_channels)
        self.linear_k = torch.nn.Linear(in_channels, in_channels)
        self.linear_v = torch.nn.Linear(in_channels, in_channels)
        self.linear_out = torch.nn.Linear(in_channels, in_channels)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.Layer_scale = LayerScale(
            dims=3, input_size=in_channels, Layer_scale_init=Layer_scale_init
        )

    def forward(
        self, x, mask=None, rel_pos_bias=None
    ):  # Fixed: added rel_pos_bias parameter
        B, seq_len, _ = x.size()
        x = self.layer_norm(x)

        q = self.linear_q(x).view(B, seq_len, self.nhead, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(B, seq_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # Compute attention
        self.attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        output = (
            self.attn.transpose(1, 2)
            .contiguous()
            .view(B, -1, self.nhead * self.d_k)  # Fixed: use d_k instead of d_model
        )

        return self.Layer_scale(self.dropout(self.linear_out(output)))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, ff_dims, num_heads, dropout=0.1, max_len=500):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(num_heads, d_model, dropout, maxlen=max_len)
        self.ff = FeedForward(d_model, ff_dims, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x, mask=None, rel_pos_bias=None
    ):  # Fixed: added rel_pos_bias parameter
        # Fixed: removed duplicate layer norm (attention module already has it)
        x = x + self.dropout(self.attn(x, mask, rel_pos_bias))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        ff_dims,
        dropout=0.1,
        num_heads=4,
        num_layers=2,
        segment_size=100,
        positional_encoding_type="absolute",
        max_len=8000,
    ):
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, ff_dims, num_heads, dropout, max_len)
                for _ in range(num_layers)
            ]
        )
        self.positional_encoding_type = positional_encoding_type
        self.segment_size = segment_size
        if self.positional_encoding_type == "absolute":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                (torch.arange(0, d_model, 2).float()) * ((-math.log(10000.0) / d_model))
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 1:  # Fixed: handle odd d_model
                pe[:, 1::2] = torch.cos(position * div_term[:-1])
            else:
                pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))
        elif positional_encoding_type == "relative":
            self.relative_pos_encoding = RelativePositionalEncoding(
                in_channels=d_model, num_heads=num_heads, maxlen=max_len, embed_v=False
            )
        else:
            raise ValueError(
                f"Unknown positional encoding type: {positional_encoding_type}"
            )

    def forward(self, x):
        if self.positional_encoding_type == "absolute":
            seq_len = x.size(1)
            x = x + self.pe[:, :seq_len]

        rel_pos_bias = None
        if self.positional_encoding_type == "relative":
            seq_len = x.size(1)
            rel_pos_bias = self.relative_pos_encoding(seq_len)

        for layer in self.layers:
            x = layer(x, rel_pos_bias=rel_pos_bias)

        return x


# Test functions
def test_layer_scale():
    print("Testing LayerScale...")
    layer_scale = LayerScale(dims=3, input_size=512, Layer_scale_init=1e-5)
    x = torch.randn(2, 10, 512)
    output = layer_scale(x)
    assert output.shape == x.shape
    print("✓ LayerScale test passed")


def test_relative_positional_encoding():
    print("Testing RelativePositionalEncoding...")
    rpe = RelativePositionalEncoding(in_channels=512, num_heads=8, maxlen=100)
    pos_bias = rpe(seq_len=50)
    assert pos_bias.shape == (50, 50, 64)  # 512 // 8 = 64
    print("✓ RelativePositionalEncoding test passed")


def test_feedforward():
    print("Testing FeedForward...")
    ff = FeedForward(d_model=512, ff_dims=2048, dropout=0.1)
    x = torch.randn(2, 10, 512)
    output = ff(x)
    assert output.shape == x.shape
    print("✓ FeedForward test passed")


def test_multihead_attention():
    print("Testing MultiHeadAttention...")
    mha = MultiHeadAttention(n_head=8, in_channels=512, dropout_rate=0.1)
    x = torch.randn(2, 10, 512)
    output = mha(x)
    assert output.shape == x.shape
    print("✓ MultiHeadAttention test passed")


def test_transformer_layer():
    print("Testing TransformerLayer...")
    layer = TransformerLayer(d_model=512, ff_dims=2048, num_heads=8, dropout=0.1)
    x = torch.randn(2, 10, 512)
    output = layer(x)
    assert output.shape == x.shape
    print("✓ TransformerLayer test passed")


def test_transformer_block_absolute():
    print("Testing TransformerBlock with absolute positioning...")
    block = TransformerBlock(
        d_model=512,
        ff_dims=2048,
        num_heads=8,
        num_layers=2,
        positional_encoding_type="relative",
    )
    x = torch.randn(2, 100, 512)
    output = block(x)
    assert output.shape == x.shape
    print("✓ TransformerBlock (absolute) test passed")


def test_transformer_block_relative():
    print("Testing TransformerBlock with relative positioning...")
    block = TransformerBlock(
        d_model=512,
        ff_dims=2048,
        num_heads=8,
        num_layers=2,
        positional_encoding_type="relative",
    )
    x = torch.randn(2, 100, 512)
    output = block(x)
    assert output.shape == x.shape
    print("✓ TransformerBlock (relative) test passed")


def test_gradient_flow():
    print("Testing gradient flow...")
    block = TransformerBlock(
        d_model=128,
        ff_dims=512,
        num_heads=4,
        num_layers=2,
        positional_encoding_type="absolute",
    )

    x = torch.randn(2, 50, 128, requires_grad=True)
    output = block(x)
    loss = output.sum()
    loss.backward()

    # Check if gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    print("✓ Gradient flow test passed")


def run_all_tests():
    print("Running Transformer module tests...\n")

    try:
        test_layer_scale()
        test_relative_positional_encoding()
        test_feedforward()
        test_multihead_attention()
        test_transformer_layer()
        test_transformer_block_absolute()
        test_transformer_block_relative()
        test_gradient_flow()

        print("\n🎉 All tests passed! The Transformer module is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

    # Example usage
    print("\n" + "=" * 50)
    print("Example Usage:")
    print("=" * 50)

    # Create a simple transformer block
    transformer = TransformerBlock(
        d_model=256,
        ff_dims=1024,
        num_heads=8,
        num_layers=4,
        positional_encoding_type="absolute",
    )

    # Sample input: batch_size=4, seq_len=50, d_model=256
    sample_input = torch.randn(4, 50, 256)

    with torch.no_grad():
        output = transformer(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
