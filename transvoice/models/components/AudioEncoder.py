import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L = L  # Encoding Compression
        self.N = N  # Output size of the latent Representation

        # setting 50% overlap
        self.conv = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)
        mixture_w = F.relu(self.conv(mixture))
        return mixture_w


