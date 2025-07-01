import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

EPS = 1e-8


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
        self.embed_v = embed_v
        self.pe_v = (
            nn.Embedding(num_embeddings=2 * maxlen, embedding_dim=self.embedding_dim)
            if embed_v
            else None
        )
        # Remove the projection layer - it's not needed for relative positioning
        # self.proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.normal_(self.pe_k.weight, mean=0.0, std=0.02)
        if self.pe_v is not None:
            nn.init.normal_(self.pe_v.weight, mean=0.0, std=0.02)

    def forward(self, seq_len):
        device = self.pe_k.weight.device
        range_vec = torch.arange(seq_len, device=device, dtype=torch.long)
        rel_mat = range_vec[None, :] - range_vec[:, None]
        rel_mat = torch.clamp(rel_mat, -self.maxlen, self.maxlen - 1)
        rel_mat = rel_mat + self.maxlen
        pe_k_output = self.pe_k(rel_mat)  # (seq_len, seq_len, embedding_dim)
        pe_k_output = torch.clamp(pe_k_output, min=-5, max=5)
        # Remove projection - direct output
        # pe_k_output = self.proj(pe_k_output)
        pe_v_output = self.pe_v(rel_mat) if self.embed_v else None
        if pe_v_output is not None:
            pe_v_output = torch.clamp(pe_v_output, min=-5, max=5)
        return pe_k_output, pe_v_output


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
    """
    Multi-Head Attention layer with relative positional encoding support.
        :param int n_head: the number of heads
        :param int in_channels: the number of features
        :param float dropout_rate: dropout rate
    """

    def __init__(
        self,
        n_head: int,
        in_channels: int,
        dropout_rate: float,
        Layer_scale_init=1.0e-5,
    ):
        super().__init__()
        assert in_channels % n_head == 0
        self.d_k = in_channels // n_head  # We assume d_v always equals d_k
        self.n_head = n_head
        self.in_channels = in_channels
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

    def forward(self, x, pos_k=None, mask=None):
        """
        Compute 'Scaled Dot Product Attention' with optional relative positional encoding.
            :param torch.Tensor x: input tensor (batch, time, d_model)
            :param torch.Tensor pos_k: relative positional encoding for keys (seq_len, seq_len, d_k)
            :param torch.Tensor mask: attention mask (batch, time1, time2)
            :return torch.Tensor: attended and transformed `value` (batch, time, d_model)
        """
        B, T, _ = x.size()
        x_norm = self.layer_norm(x)

        # Linear projections
        q = (
            self.linear_q(x_norm).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        )  # (b, h, t, d_k)
        k = (
            self.linear_k(x_norm).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        )  # (b, h, t , d_k)
        v = (
            self.linear_v(x_norm).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        )  # (b, h, t, d_k)

        # Compute content-content attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_k
        )  # (b, h, t, t)

        # Add relative positional bias if provided
        if pos_k is not None:
            # pos_k shape: (seq_len, seq_len, d_k)
            # Ensure pos_k matches the current sequence length
            if pos_k.size(0) != T or pos_k.size(1) != T:
                # If pos_k doesn't match sequence length, we need to slice or pad
                if pos_k.size(0) >= T and pos_k.size(1) >= T:
                    # Slice to match sequence length
                    pos_k = pos_k[:T, :T, :]
                else:
                    # This shouldn't happen if relative encoding is generated correctly
                    raise ValueError(
                        f"pos_k shape {pos_k.shape} doesn't match sequence length {T}"
                    )

            # Compute relative position scores
            # pos_k: (seq_len, seq_len, d_k)
            # q: (b, h, seq_len, d_k)
            # We need to compute einsum('bhid,ijd->bhij', q, pos_k)

            # Method 1: Using einsum (cleaner)
            rel_scores = torch.einsum("bhid,ijd->bhij", q, pos_k)  # (b, h, t, t)

            # Add relative position bias to content scores
            scores = scores + (rel_scores / math.sqrt(self.d_k))

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # (batch, time1, time2) -> (batch, 1, time1, time2)
                mask = mask.unsqueeze(1)
            mask = mask.eq(0)  # Convert to boolean mask where 0s become True
            min_value = float(
                np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)

        # Apply softmax
        self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        # Apply mask to attention weights if provided
        if mask is not None:
            self.attn = self.attn.masked_fill(mask, 0.0)

        # Apply dropout to attention weights
        p_attn = self.dropout(self.attn)

        # Apply attention to values
        x_attended = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)

        # Concatenate heads
        x_attended = (
            x_attended.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_k)
        )  # (batch, time1, d_model)

        # Final linear projection with layer scale and dropout
        output = self.linear_out(x_attended)
        return self.Layer_scale(self.dropout(output))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, ff_dims, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(num_heads, d_model, dropout)
        self.ff = FeedForward(d_model, ff_dims, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, pos_k=None):
        # Pre-norm architecture: attention already handles normalization internally
        x = x + self.attn(x, pos_k=pos_k, mask=mask)
        x = x + self.dropout(self.ff(self.norm2(x)))
        if torch.isnan(x).any():
            raise ValueError("NaN detected in transformer layer")
        return x


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        """
        Args:
            d_model (int): Embedding dimension (model size)
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Output tensor with positional encoding added
        """
        _, seq_len, _ = x.size()

        # Compute position indices and scaling factors
        position = torch.arange(
            seq_len, dtype=torch.float32, device=x.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=x.device)
            * (-math.log(10000.0) / self.d_model)
        )

        # Create positional encoding
        pe = torch.zeros((1, seq_len, self.d_model), device=x.device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 == 1:
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)

        pe = pe.expand(x.size(0), -1, -1)  # (B, seq_len, d_model)
        return x + pe


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        ff_dims,
        dropout=0.1,
        num_heads=4,
        num_layers=2,
        positional_encoding_type="absolute",
        max_len=8000,
        n_sources=2,
        freq_dim=None,
    ):
        super(TransformerBlock, self).__init__()

        self.frontend = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.GroupNorm(1, d_model),
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, ff_dims, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.positional_encoding_type = positional_encoding_type
        if self.positional_encoding_type == "absolute":
            self.absolute_pos_encoding = AbsolutePositionalEncoding(d_model=d_model)

        elif positional_encoding_type == "relative":
            self.relative_pos_encoding = RelativePositionalEncoding(
                in_channels=d_model, num_heads=num_heads, maxlen=max_len, embed_v=False
            )
        else:
            raise ValueError(
                f"Unknown positional encoding type: {positional_encoding_type}"
            )
        self.n_sources = n_sources
        if freq_dim is not None:
            self.mask_estim = nn.Linear(d_model, n_sources * freq_dim)

    def forward(self, x):

        if x.dim() == 3 and x.size(1) == 1:
            # Assume input is raw audio: (B, 1, L)
            x = self.frontend(x)  # (B, D, T)
            x = x.transpose(1, 2)  # (B, T, D)

        # Add positional encodings
        if self.positional_encoding_type == "absolute":
            x = self.absolute_pos_encoding(x)

        pos_k = None
        if self.positional_encoding_type == "relative":
            seq_len = x.size(1)
            pos_k, _ = self.relative_pos_encoding(seq_len)  # Get only pos_k

        for layer in self.layers:
            x = layer(x, pos_k=pos_k)

        if hasattr(self, "mask_estim"):
            masks = self.mask_estim(x)  # (B, T, n_sources * F)
            B, T, _ = masks.shape
            F = masks.shape[-1] // self.n_sources
            masks = masks.view(B, T, self.n_sources, F)
            masks = torch.sigmoid(masks)
            return masks
        else:
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
    pos_k, pos_v = rpe(seq_len=50)
    expected_dim = 512 // 8  # d_k = 64
    print(f"  pos_k shape: {pos_k.shape}, expected: (50, 50, {expected_dim})")
    assert pos_k.shape == (
        50,
        50,
        expected_dim,
    ), f"Expected (50, 50, {expected_dim}), got {pos_k.shape}"
    assert pos_v is None  # embed_v=False
    print("✓ RelativePositionalEncoding test passed")


def test_relative_positional_encoding_with_v():
    print("Testing RelativePositionalEncoding with embed_v=True...")
    rpe = RelativePositionalEncoding(
        in_channels=512, num_heads=8, maxlen=100, embed_v=True
    )
    pos_k, pos_v = rpe(seq_len=30)
    expected_dim = 512 // 8  # d_k = 64
    assert pos_k.shape == (30, 30, expected_dim)
    assert pos_v.shape == (30, 30, expected_dim)
    print("✓ RelativePositionalEncoding with embed_v test passed")


def test_feedforward():
    print("Testing FeedForward...")
    ff = FeedForward(d_model=512, ff_dims=2048, dropout=0.1)
    x = torch.randn(2, 10, 512)
    output = ff(x)
    assert output.shape == x.shape
    print("✓ FeedForward test passed")


def test_multihead_attention():
    print("Testing MultiHeadAttention without relative positioning...")
    mha = MultiHeadAttention(n_head=8, in_channels=512, dropout_rate=0.1)
    x = torch.randn(2, 10, 512)
    output = mha(x)
    assert output.shape == x.shape
    print("✓ MultiHeadAttention test passed")


def test_multihead_attention_with_relative_pos():
    print("Testing MultiHeadAttention with relative positioning...")
    mha = MultiHeadAttention(n_head=8, in_channels=512, dropout_rate=0.1)
    rpe = RelativePositionalEncoding(in_channels=512, num_heads=8, maxlen=100)

    x = torch.randn(2, 10, 512)
    pos_k, _ = rpe(seq_len=10)

    print(f"  Input x shape: {x.shape}")
    print(f"  pos_k shape: {pos_k.shape}")
    print(f"  Expected d_k: {512 // 8}")

    output = mha(x, pos_k=pos_k)
    assert output.shape == x.shape
    print("✓ MultiHeadAttention with relative positioning test passed")


def test_multihead_attention_with_mask():
    print("Testing MultiHeadAttention with mask...")
    mha = MultiHeadAttention(n_head=8, in_channels=512, dropout_rate=0.1)

    x = torch.randn(2, 10, 512)
    # Create a simple causal mask
    mask = torch.tril(torch.ones(2, 10, 10))  # Lower triangular mask

    output = mha(x, mask=mask)
    assert output.shape == x.shape
    print("✓ MultiHeadAttention with mask test passed")


def test_multihead_attention_comprehensive():
    print("Testing MultiHeadAttention with relative positioning and mask...")
    mha = MultiHeadAttention(n_head=8, in_channels=512, dropout_rate=0.1)
    rpe = RelativePositionalEncoding(in_channels=512, num_heads=8, maxlen=100)

    x = torch.randn(2, 10, 512)
    pos_k, _ = rpe(seq_len=10)
    mask = torch.tril(torch.ones(2, 10, 10))  # Causal mask

    output = mha(x, pos_k=pos_k, mask=mask)
    assert output.shape == x.shape

    # Check that attention weights are properly masked
    assert mha.attn is not None
    assert mha.attn.shape == (2, 8, 10, 10)  # (batch, heads, seq_len, seq_len)

    # Verify causal masking: upper triangular should be zero
    for b in range(2):
        for h in range(8):
            upper_tri = torch.triu(mha.attn[b, h], diagonal=1)
            assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6)

    print("✓ MultiHeadAttention comprehensive test passed")


def test_transformer_layer():
    print("Testing TransformerLayer...")
    layer = TransformerLayer(d_model=512, ff_dims=2048, num_heads=8, dropout=0.1)
    x = torch.randn(2, 10, 512)
    output = layer(x)
    assert output.shape == x.shape
    print("✓ TransformerLayer test passed")


def test_transformer_layer_with_relative_pos():
    print("Testing TransformerLayer with relative positioning...")
    layer = TransformerLayer(d_model=512, ff_dims=2048, num_heads=8, dropout=0.1)
    rpe = RelativePositionalEncoding(in_channels=512, num_heads=8, maxlen=100)

    x = torch.randn(2, 10, 512)
    pos_k, _ = rpe(seq_len=10)

    output = layer(x, pos_k=pos_k)
    assert output.shape == x.shape
    print("✓ TransformerLayer with relative positioning test passed")


def test_transformer_block_absolute():
    print("Testing TransformerBlock with absolute positioning...")
    block = TransformerBlock(
        d_model=512,
        ff_dims=2048,
        num_heads=8,
        num_layers=2,
        positional_encoding_type="absolute",
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


def test_relative_positioning_benefits():
    print("Testing that relative positioning produces different outputs...")

    # Create identical inputs
    x = torch.randn(1, 20, 256)

    # Test without relative positioning
    mha_no_rel = MultiHeadAttention(n_head=8, in_channels=256, dropout_rate=0.0)
    mha_no_rel.eval()  # Disable dropout for consistent results

    # Test with relative positioning
    mha_with_rel = MultiHeadAttention(n_head=8, in_channels=256, dropout_rate=0.0)
    mha_with_rel.eval()
    rpe = RelativePositionalEncoding(in_channels=256, num_heads=8, maxlen=50)

    # Make weights identical for fair comparison
    with torch.no_grad():
        for param1, param2 in zip(mha_no_rel.parameters(), mha_with_rel.parameters()):
            param2.copy_(param1)

    with torch.no_grad():
        output_no_rel = mha_no_rel(x)
        pos_k, _ = rpe(seq_len=20)
        output_with_rel = mha_with_rel(x, pos_k=pos_k)

    # Outputs should be different due to relative positioning
    assert not torch.allclose(
        output_no_rel, output_with_rel, atol=1e-5
    ), "Relative positioning should produce different outputs"

    print("✓ Relative positioning produces different outputs as expected")


def test_gradient_flow():
    print("Testing gradient flow...")
    block = TransformerBlock(
        d_model=128,
        ff_dims=512,
        num_heads=4,
        num_layers=2,
        positional_encoding_type="relative",
    )

    x = torch.randn(2, 50, 128, requires_grad=True)
    output = block(x)
    loss = output.sum()
    loss.backward()

    # Check if gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape

    # Check if model parameters have gradients
    for name, param in block.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"

    print("✓ Gradient flow test passed")


def run_all_tests():
    print("Running Transformer module tests...\n")

    try:
        test_layer_scale()
        test_relative_positional_encoding()
        test_relative_positional_encoding_with_v()
        test_feedforward()
        test_multihead_attention()
        test_multihead_attention_with_relative_pos()
        test_multihead_attention_with_mask()
        test_multihead_attention_comprehensive()
        test_transformer_layer()
        test_transformer_layer_with_relative_pos()
        test_transformer_block_absolute()
        test_transformer_block_relative()
        test_relative_positioning_benefits()
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
        positional_encoding_type="relative",
    )

    # Sample input: batch_size=4, seq_len=50, d_model=256
    sample_input = torch.randn(4, 50, 256)

    with torch.no_grad():
        output = transformer(sample_input)
        print(f"Input shape: {sample_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Parameters: {sum(p.numel() for p in transformer.parameters()):,}")

    print("\n" + "=" * 50)
    print("Attention Pattern Visualization:")
    print("=" * 50)

    # Show attention patterns with relative positioning
    transformer.eval()
    with torch.no_grad():
        # Process a small sequence to see attention patterns
        small_input = torch.randn(1, 10, 256)
        _ = transformer(small_input)

        # Get attention weights from the first layer
        first_layer_attn = transformer.layers[0].attn.attn
        if first_layer_attn is not None:
            print(f"Attention weights shape: {first_layer_attn.shape}")
            print("First head attention pattern:")
            attn_pattern = first_layer_attn[0, 0].numpy()  # First batch, first head
            for i in range(min(5, attn_pattern.shape[0])):
                row = " ".join(
                    [
                        f"{val:.3f}"
                        for val in attn_pattern[i, : min(5, attn_pattern.shape[1])]
                    ]
                )
                print(f"  Pos {i}: {row}")
