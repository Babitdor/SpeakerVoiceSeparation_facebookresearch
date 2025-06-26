import torch
from Transwave import Transwave


def main():
    # Model hyperparameters (example values, adjust as needed)
    N = 64  # Number of filters in autoencoder
    L = 16  # Length of the filters (in samples)
    H = 128  # Hidden size in DPRNN/Transformer
    R = 2  # Number of dual-path blocks
    C = 2  # Number of speakers to separate
    sr = 8000  # Sample rate
    segment = 100  # Segment size (ms or samples, as used in your code)
    input_normalize = True
    nheads = 4  # Number of attention heads

    # Instantiate the model
    model = Transwave(N, L, H, R, C, sr, segment, input_normalize, nheads)

    # Dummy input: batch_size=2, input_length=32000 (2 seconds at 16kHz)
    batch_size = 2
    input_length = 32000
    dummy_input = torch.randn(batch_size, input_length)

    # Forward pass
    output = model(dummy_input)
    print(
        "Model output shape:", output.shape
    )  # Should be (R, batch_size, C, input_length)

    # Dummy target for loss (same shape as output)
    dummy_target = torch.randn_like(output)

    # Example loss (MSE)
    criterion = torch.nn.MSELoss()
    loss = criterion(output, dummy_target)
    print("Dummy loss:", loss.item())

    # Backward pass
    loss.backward()
    print("Backward pass successful")

    # Optionally, print model summary
    print(model)


if __name__ == "__main__":
    main()
