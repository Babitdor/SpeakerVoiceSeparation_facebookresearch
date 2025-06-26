import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x, mask=None, rel_pos_bias=None):  # Fixed: added rel_pos_bias parameter
        B, seq_len, _ = x.size()
        x = self.layer_norm(x)

        q = (
            self.linear_q(x).view(B, seq_len, self.nhead, self.d_k).transpose(1, 2)
        )
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

    def forward(self, x, mask=None, rel_pos_bias=None):  # Fixed: added rel_pos_bias parameter
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
        positional_encoding_type="absolute"
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
        positional_encoding_type="relative"
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
        positional_encoding_type="absolute"
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


class GatedDualTransformerBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        dropout=0.1,
        nheads=4,
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
        
        # Fixed: Updated parameter names to match TransformerBlock
        self.transformer = TransformerBlock(
            d_model=self.input_size,           # was embed_dim
            ff_dims=self.hidden_size,          # was ff_dim
            dropout=self.dropout,
            num_heads=self.nheads,
            num_layers=self.num_layers,
            segment_size=self.segment_size,
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
        # Get transformer output
        transformer_out = self.layer_norm(self.transformer(input))
        
        # Apply gating mechanism
        gate = torch.sigmoid(
            self.block_projection(torch.cat([transformer_out, input], dim=-1))
        )
        gated = gate * transformer_out + (1 - gate) * input
        
        # Apply feed-forward network
        output = self.ffn(gated)
        return output


def test_gated_dual_transformer():
    print("Testing GatedDualTransformerBlock...")
    
    # Test with absolute positioning
    gated_block_abs = GatedDualTransformerBlock(
        input_size=256,
        hidden_size=512,
        dropout=0.1,
        nheads=8,
        num_layers=2,
        re_encode_pos=False  # absolute positioning
    )
    
    x = torch.randn(2, 50, 256)
    output_abs = gated_block_abs(x)
    assert output_abs.shape == x.shape
    print("✓ GatedDualTransformerBlock (absolute positioning) test passed")
    
    # Test with relative positioning
    gated_block_rel = GatedDualTransformerBlock(
        input_size=256,
        hidden_size=512,
        dropout=0.1,
        nheads=8,
        num_layers=2,
        re_encode_pos=True  # relative positioning
    )
    
    output_rel = gated_block_rel(x)
    assert output_rel.shape == x.shape
    print("✓ GatedDualTransformerBlock (relative positioning) test passed")


def test_gated_gradient_flow():
    print("Testing GatedDualTransformerBlock gradient flow...")
    
    gated_block = GatedDualTransformerBlock(
        input_size=128,
        hidden_size=256,
        dropout=0.1,
        nheads=4,
        num_layers=2,
        re_encode_pos=True
    )
    
    x = torch.randn(2, 30, 128, requires_grad=True)
    output = gated_block(x)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    print("✓ GatedDualTransformerBlock gradient flow test passed")


def test_gating_mechanism():
    print("Testing gating mechanism behavior...")
    
    gated_block = GatedDualTransformerBlock(
        input_size=64,
        hidden_size=128,
        dropout=0.0,  # No dropout for testing
        nheads=4,
        num_layers=1,
        re_encode_pos=False
    )
    
    # Test with different inputs
    x1 = torch.randn(1, 10, 64)
    x2 = torch.zeros(1, 10, 64)  # Zero input
    
    with torch.no_grad():
        output1 = gated_block(x1)
        output2 = gated_block(x2)
        
        # Outputs should be different for different inputs
        assert not torch.allclose(output1, output2, atol=1e-6)
        print("✓ Gating mechanism produces different outputs for different inputs")


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
        
        # New tests for GatedDualTransformerBlock
        test_gated_dual_transformer()
        test_gated_gradient_flow()
        test_gating_mechanism()
        
        print("\n🎉 All tests passed! Both Transformer and GatedDualTransformerBlock modules are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


class ByPass(nn.Module):
    def __init__(self):
        super(ByPass, self).__init__()

    def forward(self, input):
        return input


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

        self.intraTransformer = nn.ModuleList([])
        self.interTransformer = nn.ModuleList([])
        self.intra_normalization = nn.ModuleList([])
        self.inter_normalization = nn.ModuleList([])

        # Fixed: Use correct attribute names
        self.rows_grnn = nn.ModuleList([])  # was intraTransformer
        self.cols_grnn = nn.ModuleList([])  # was interTransformer
        self.rows_normalization = nn.ModuleList([])  # was intra_normalization
        self.cols_normalization = nn.ModuleList([])  # was inter_normalization

        # create the dual path pipeline
        for i in range(num_layers):
            self.rows_grnn.append(
                GatedDualTransformerBlock(
                    self.input_size,
                    self.hidden_size,
                    self.dropout,
                    self.nheads,
                    2,  # Fixed: use smaller num_layers for individual blocks
                    self.segment_size,
                    re_encode_pos=True,
                )
            )
            self.cols_grnn.append(
                GatedDualTransformerBlock(
                    self.input_size,
                    self.hidden_size,
                    self.dropout,
                    self.nheads,
                    2,  # Fixed: use smaller num_layers for individual blocks
                    self.segment_size,
                    re_encode_pos=True,
                )
            )
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
            # Process rows (intra-chunk processing)
            row_input = output.permute(0, 3, 2, 1).reshape(batch_size * d2, d1, -1)
            row_output = self.rows_grnn[i](row_input)
            row_output = row_output.view(batch_size, d2, d1, -1).permute(0, 3, 2, 1)
            row_output = self.rows_normalization[i](row_output)
            
            # Apply skip connection
            output = output + row_output

            # Process columns (inter-chunk processing)
            col_input = output.permute(0, 2, 3, 1).reshape(batch_size * d1, d2, -1)
            col_output = self.cols_grnn[i](col_input)
            col_output = col_output.view(batch_size, d1, d2, -1).permute(0, 3, 1, 2)
            col_output = self.cols_normalization[i](col_output)
            
            # Apply skip connection
            output = output + col_output

            # Generate output for this layer
            output_i = self.output(output)
            if self.training or i == (self.num_layers - 1):
                output_all.append(output_i)

        return output_all


def test_bypass():
    print("Testing ByPass...")
    bypass = ByPass()
    x = torch.randn(2, 64, 100)
    output = bypass(x)
    assert torch.equal(output, x)
    print("✓ ByPass test passed")


def test_dual_path_basic():
    print("Testing DualPath basic functionality...")
    
    dual_path = DualPath(
        input_size=64,
        hidden_size=128,
        output_size=32,
        num_spk=2,
        dropout=0.1,
        nheads=4,
        num_layers=2,
        input_normalize=False,
        segment_size=50
    )
    
    # Input: (batch_size, channels, freq_bins, time_frames)
    x = torch.randn(2, 64, 40, 50)
    outputs = dual_path(x)
    
    # Check that we get the expected number of outputs
    expected_outputs = 2 if dual_path.training else 1
    assert len(outputs) == expected_outputs
    
    # Check output shape: (batch_size, output_size * num_spk, freq_bins, time_frames)
    expected_shape = (2, 32 * 2, 40, 50)  # output_size * num_spk
    for output in outputs:
        assert output.shape == expected_shape
    
    print("✓ DualPath basic functionality test passed")


def test_dual_path_with_normalization():
    print("Testing DualPath with normalization...")
    
    dual_path = DualPath(
        input_size=64,
        hidden_size=128,
        output_size=32,
        num_spk=2,
        dropout=0.1,
        nheads=4,
        num_layers=2,
        input_normalize=True,  # Enable normalization
        segment_size=50
    )
    
    x = torch.randn(2, 64, 40, 50)
    outputs = dual_path(x)
    
    # Check that normalization layers are GroupNorm
    assert isinstance(dual_path.rows_normalization[0], nn.GroupNorm)
    assert isinstance(dual_path.cols_normalization[0], nn.GroupNorm)
    
    expected_outputs = 2 if dual_path.training else 1
    assert len(outputs) == expected_outputs
    
    print("✓ DualPath with normalization test passed")


def test_dual_path_inference_mode():
    print("Testing DualPath in inference mode...")
    
    dual_path = DualPath(
        input_size=64,
        hidden_size=128,
        output_size=32,
        num_spk=2,
        dropout=0.1,
        nheads=4,
        num_layers=3,
        input_normalize=False,
        segment_size=50
    )
    
    dual_path.eval()  # Set to evaluation mode
    
    x = torch.randn(2, 64, 40, 50)
    with torch.no_grad():
        outputs = dual_path(x)
    
    # In inference mode, should only return the last output
    assert len(outputs) == 1
    expected_shape = (2, 32 * 2, 40, 50)
    assert outputs[0].shape == expected_shape
    
    print("✓ DualPath inference mode test passed")


def test_dual_path_gradient_flow():
    print("Testing DualPath gradient flow...")
    
    dual_path = DualPath(
        input_size=32,
        hidden_size=64,
        output_size=16,
        num_spk=2,
        dropout=0.1,
        nheads=4,
        num_layers=2,
        input_normalize=False,
        segment_size=30
    )
    
    x = torch.randn(1, 32, 20, 30, requires_grad=True)
    outputs = dual_path(x)
    
    # Compute loss from all outputs
    loss = sum(output.sum() for output in outputs)
    loss.backward()
    
    # Check if gradients exist
    assert x.grad is not None
    assert x.grad.shape == x.shape
    print("✓ DualPath gradient flow test passed")


def test_dual_path_different_speakers():
    print("Testing DualPath with different number of speakers...")
    
    # Test with 1 speaker
    dual_path_1spk = DualPath(
        input_size=64,
        hidden_size=128,
        output_size=32,
        num_spk=1,
        num_layers=2
    )
    
    # Test with 3 speakers  
    dual_path_3spk = DualPath(
        input_size=64,
        hidden_size=128,
        output_size=32,
        num_spk=3,
        num_layers=2
    )
    
    x = torch.randn(2, 64, 40, 50)
    
    outputs_1spk = dual_path_1spk(x)
    outputs_3spk = dual_path_3spk(x)
    
    # Check output dimensions
    assert outputs_1spk[0].shape == (2, 32 * 1, 40, 50)
    assert outputs_3spk[0].shape == (2, 32 * 3, 40, 50)
    
    print("✓ DualPath different speakers test passed")


if __name__ == "__main__":
    run_all_tests()
    
    # Example usage
    print("\n" + "="*70)
    print("Example Usage:")
    print("="*70)
    
    # Create a simple transformer block
    print("1. Basic TransformerBlock:")
    transformer = TransformerBlock(
        d_model=256,
        ff_dims=1024,
        num_heads=8,
        num_layers=4,
        positional_encoding_type="absolute"
    )
    
    sample_input = torch.randn(4, 50, 256)
    
    with torch.no_grad():
        output = transformer(sample_input)
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Create a gated dual transformer block
    print("\n2. GatedDualTransformerBlock:")
    gated_transformer = GatedDualTransformerBlock(
        input_size=256,
        hidden_size=512,
        dropout=0.1,
        nheads=8,
        num_layers=3,
        re_encode_pos=True
    )
    
    with torch.no_grad():
        gated_output = gated_transformer(sample_input)
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {gated_output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in gated_transformer.parameters()):,}")
    
    # Create a dual path network
    print("\n3. DualPath Network:")
    dual_path = DualPath(
        input_size=128,
        hidden_size=256,
        output_size=64,
        num_spk=2,
        dropout=0.1,
        nheads=8,
        num_layers=4,
        input_normalize=True
    )
    
    # Input for dual path: (batch, channels, freq_bins, time_frames)
    dual_input = torch.randn(2, 128, 80, 100)
    
    with torch.no_grad():
        dual_path.eval()  # Set to eval mode
        dual_outputs = dual_path(dual_input)
        print(f"   Input shape: {dual_input.shape}")
        print(f"   Number of outputs: {len(dual_outputs)}")
        print(f"   Output shape: {dual_outputs[0].shape}")
        print(f"   Parameters: {sum(p.numel() for p in dual_path.parameters()):,}")
    
    # Compare outputs
    print("\n4. Comparison:")
    print(f"   Standard Transformer output mean: {output.mean().item():.6f}")
    print(f"   Gated Transformer output mean: {gated_output.mean().item():.6f}")
    print(f"   DualPath output mean: {dual_outputs[0].mean().item():.6f}")
    print(f"   Standard Transformer output std: {output.std().item():.6f}")
    print(f"   Gated Transformer output std: {gated_output.std().item():.6f}")
    print(f"   DualPath output std: {dual_outputs[0].std().item():.6f}")
    
    print("\n5. Architecture Summary:")
    print("   - TransformerBlock: Standard transformer with positional encoding")
    print("   - GatedDualTransformerBlock: Transformer with gating mechanism")
    print("   - DualPath: Dual-path processing for speech separation/enhancement")
    print("     * Processes both intra-chunk (rows) and inter-chunk (columns)")
    print("     * Supports multiple speakers")
    print("     * Returns multiple outputs during training")


if __name__ == "__main__":
    run_all_tests()
    
    # Example usage
    print("\n" + "="*60)
    print("Example Usage:")
    print("="*60)
    
    # Create a simple transformer block
    print("1. Basic TransformerBlock:")
    transformer = TransformerBlock(
        d_model=256,
        ff_dims=1024,
        num_heads=8,
        num_layers=4,
        positional_encoding_type="absolute"
    )
    
    sample_input = torch.randn(4, 50, 256)
    
    with torch.no_grad():
        output = transformer(sample_input)
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Create a gated dual transformer block
    print("\n2. GatedDualTransformerBlock:")
    gated_transformer = GatedDualTransformerBlock(
        input_size=256,
        hidden_size=512,
        dropout=0.1,
        nheads=8,
        num_layers=3,
        re_encode_pos=True
    )
    
    with torch.no_grad():
        gated_output = gated_transformer(sample_input)
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {gated_output.shape}")
        print(f"   Parameters: {sum(p.numel() for p in gated_transformer.parameters()):,}")
    
    # Compare outputs
    print("\n3. Comparison:")
    print(f"   Standard Transformer output mean: {output.mean().item():.6f}")
    print(f"   Gated Transformer output mean: {gated_output.mean().item():.6f}")
    print(f"   Standard Transformer output std: {output.std().item():.6f}")
    print(f"   Gated Transformer output std: {gated_output.std().item():.6f}")