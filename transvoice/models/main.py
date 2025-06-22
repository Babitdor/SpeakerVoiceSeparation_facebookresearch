import torch
import torch.nn.functional as F
import numpy as np
from transwave import SWave


def main():
    # Example hyperparameters (adjust as needed)
    N = 64  # Feature dim
    nheads = 8
    L = 16  # Encoder kernel size
    H = 128  # Hidden dim in transformer
    R = 4  # Number of dual-path layers
    C = 2  # Number of speakers
    sr = 8000  # Sample rate
    segment = 4  # Segment length in seconds
    input_normalize = False

    # Create model
    model = SWave(N, L, H, R, C, sr, segment, input_normalize, nheads=nheads)

    # Generate example input: two "speakers" as sine waves at different frequencies
    batch_size = 1
    audio_len = sr * segment  # 4 seconds
    t = np.linspace(0, segment, audio_len, endpoint=False)
    speaker1 = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz (A4)
    speaker2 = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880 Hz (A5)
    mixture = speaker1 + speaker2  # Mix the two speakers

    # Convert to torch tensor and batch it
    mixture_tensor = torch.tensor(mixture, dtype=torch.float32).unsqueeze(
        0
    )  # (1, audio_len)

    # Forward pass
    with torch.no_grad():
        separated = model(mixture_tensor)

    print("Input shape:", mixture_tensor.shape)
    print("Output shape:", separated.shape)  # (num_blocks, batch, C, T)

    # Optionally, check output stats
    print("Separated min/max:", separated.min().item(), separated.max().item())

    # Listen to separated output (requires sounddevice or IPython.display, not shown here)
    # For real audio, save with torchaudio.save or librosa.output.write_wav


if __name__ == "__main__":
    main()
