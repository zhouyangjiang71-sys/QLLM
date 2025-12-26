import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class QLLMMixer(nn.Module):
    def __init__(self):
        super(QLLMMixer, self).__init__()
    def forward(self, agents_q, global_state):
        agents_q = torch.squeeze(agents_q)
        a = agents_q.shape[0]
        b = agents_q.shape[1]
        agents_q = agents_q.reshape(-1, agents_q.shape[-1])
        global_state = global_state.reshape(-1, global_state.shape[-1])
        batch_size = agents_q.size(0)
        agents_data = global_state[:, :6].view(batch_size, 2, 3)
        food_data = global_state[:, 6:12].view(batch_size, 2, 3)
    
        agents_level = agents_data[:, :, 2]
        agents_pos = agents_data[:, :, :2].unsqueeze(2)
        food_pos = food_data[:, :, :2].unsqueeze(1)
    
        dxdy = torch.abs(agents_pos - food_pos).sum(dim=-1)
        adjacent = (dxdy == 1).float()
        active_food_mask = ((food_data[:, :, 0] != -1) & (food_data[:, :, 1] != -1)).unsqueeze(1)
        adjacent = adjacent * active_food_mask
    
        total_contribution = torch.zeros_like(agents_level)
        for j in range(2):
            adjacent_j = adjacent[:, :, j]
            food_level_j = food_data[:, j, 2]
            sum_adj_levels = (agents_level * adjacent_j).sum(dim=1)
            num_adj = adjacent_j.sum(dim=1)
            valid = ((num_adj >= 2) & (sum_adj_levels >= food_level_j)).float().unsqueeze(1)
            contribution_j = adjacent_j * food_level_j.unsqueeze(1) * valid
            total_contribution += contribution_j
    
        agent_contrib = agents_level * total_contribution
        weights = torch.softmax(agent_contrib, dim=1)
        global_q = (agents_q * weights).sum(dim=1, keepdim=True)
        global_q=global_q
        return (global_q*agents_q.shape[-1]).reshape(a, b, 1).cuda()