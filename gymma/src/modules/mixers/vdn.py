import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agents_q, global_state):
        return torch.sum(agents_q, dim=2, keepdim=True)