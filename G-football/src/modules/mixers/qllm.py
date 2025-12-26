import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class QLLMMixer(nn.Module):
    def __init__(self):
        super(QLLMMixer, self).__init__()
    def forward(self, agents_q, global_state):
        a = agents_q.shape[0]
        b = agents_q.shape[1]
        agents_q = agents_q.reshape(-1, agents_q.shape[-1])
        global_state = global_state.reshape(-1, global_state.shape[-1])

        batch_size = agents_q.size(0)
        n_agents = agents_q.size(1)

        # Extract ball information from first agent's observation
        ball_pos = global_state[:, 88:90]
        holder_info = global_state[:, 94:97]
        is_my_team = holder_info[:, 1].bool()

        # Calculate distance to goal
        goal_pos = torch.tensor([1.0, 0.0], device=agents_q.device)
        ball_goal_dist = torch.norm(ball_pos - goal_pos, dim=1)
        valid_advance = is_my_team & (ball_goal_dist > 0.19) & (ball_goal_dist < 0.99)

        # Calculate agent-ball distances and holder mask
        all_dists = []
        for i in range(n_agents):
            agent_obs = global_state[:, i * 115:(i + 1) * 115]
            agent_id = torch.argmax(agent_obs[:, 97:108], dim=1)
            pos_x = agent_obs[torch.arange(batch_size), 2 * agent_id]
            pos_y = agent_obs[torch.arange(batch_size), 2 * agent_id + 1]
            dist = torch.norm(torch.stack([pos_x - ball_pos[:, 0],
                                           pos_y - ball_pos[:, 1]], dim=1), dim=1)
            all_dists.append(dist.unsqueeze(1))

        dist_matrix = torch.cat(all_dists, dim=1)
        holder_mask = F.one_hot(torch.argmin(dist_matrix, dim=1), num_classes=n_agents).bool()

        # Calculate agent-goal distances
        goal_dists = []
        for i in range(n_agents):
            agent_obs = global_state[:, i * 115:(i + 1) * 115]
            agent_id = torch.argmax(agent_obs[:, 97:108], dim=1)
            pos_x = agent_obs[torch.arange(batch_size), 2 * agent_id]
            pos_y = agent_obs[torch.arange(batch_size), 2 * agent_id + 1]
            dist = torch.norm(torch.stack([pos_x - goal_pos[0], pos_y - goal_pos[1]], dim=1), dim=1)
            goal_dists.append(dist.unsqueeze(1))

        goal_dist_matrix = torch.cat(goal_dists, dim=1)

        # Calculate weights components
        advance_component = holder_mask.float() * valid_advance.unsqueeze(1) * 5.0
        position_component = 1.0 / (goal_dist_matrix + 1e-6)

        # Combine components with softmax
        combined = torch.where(valid_advance.unsqueeze(1), advance_component, position_component)
        weights = F.softmax(combined, dim=1)

        # Compute global Q-value
        global_q = (agents_q * weights).sum(dim=1, keepdim=True)
        return (global_q * agents_q.shape[-1]).reshape(a, b, 1).cuda()