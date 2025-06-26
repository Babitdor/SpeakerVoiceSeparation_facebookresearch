import torch
import torch.nn as nn
import torch.nn.functional as F
from transvoice.scripts.utils import overlap_and_add


class Decoder(nn.Module):
    def __init__(self, L):
        super(Decoder, self).__init__()
        self.L = L

    def forward(self, est_source):
        est_source = torch.transpose(est_source, 2, 3)
        est_source = F.avg_pool2d(est_source, (1, self.L))
        est_source = overlap_and_add(est_source, self.L // 2)
        return est_source
