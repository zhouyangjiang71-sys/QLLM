from collections.abc import Iterable
import numpy as np
import gymnasium as gym


def actions_to_onehot(num_actions, actions):
    """
    Transfer actions to onehot representation
    :param num_actions: list of number of actions of each agent
    :param actions: list of actions (int) for each agent
    :return: onehot representation of actions
    """
    onehot = [[0] * num_action for num_action in num_actions]
    for ag, act in enumerate(actions):
        onehot[ag][act] = 1
    return onehot


class MatrixGame(gym.Env):
    def __init__(self, payoff_matrix, ep_length, last_action_state=True):
        """
        Create matrix game
        :param payoff_matrix: list of lists or numpy array for payoff matrix of all agents
        :param ep_length: length of episode (before done is True)
        :param last_action_state: boolean flag indicating whether last actions should be returned
                                  as state of the environment or just 0s for all agents
        """
        self.payoff = payoff_matrix[0]
        self.n_agents = 2
        self.num_actions = [len(self.payoff) for i in range(self.n_agents)]

        self.ep_length = ep_length
        self.last_action_state = last_action_state

        self.last_actions = [-1 for _ in range(self.n_agents)]
        self.t = 0

        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(num_action) for num_action in self.num_actions]
        )

        shape = (self.n_agents,)
        low = np.zeros(shape)
        high = np.ones(shape)
        obs_space = gym.spaces.Box(shape=shape, low=low, high=high)
        self.observation_space = gym.spaces.Tuple(
            [obs_space for _ in range(self.n_agents)]
        )

    def _make_obs(self):
        return self.last_actions

    def reset(self, seed=None, options=None):
        self.t = 0
        return self._make_obs(), {}

    def step(self, action):
        self.t += 1

        self.last_actions = action
        rewards = [0 for _ in range(len(action))]
        rewards[0] = self.payoff[action[0]][action[1]] / 2
        rewards[1] = self.payoff[action[0]][action[1]] / 2
        done = self.t >= self.ep_length
        truncated = False
        return self._make_obs(), rewards, done, truncated, {}

    def get_state(self):
        return self.last_actions

    def state_size(self):
        return len(self.last_actions)

    def render(self):
        print(f"Step {self.t}:")
        for i in range(self.n_agents):
            print(f"\tAgent {i + 1} action: {self.last_actions[i]}")


class TwoStep(gym.Env):
    def __init__(self, payoff_matrix1, payoff_matrix2):
        self.payoff1 = payoff_matrix1
        self.payoff2 = payoff_matrix1

        self.matrix1 = MatrixGame(payoff_matrix1, ep_length=1, last_action_state=False)
        self.matrix2 = MatrixGame(payoff_matrix2, ep_length=1, last_action_state=False)
        self.n_agents = self.matrix1.n_agents

        self.ep_length = 2

        self.action_space = self.matrix1.action_space
        shape = (3,)
        low = np.zeros(shape)
        high = np.ones(shape)
        obs_space = gym.spaces.Box(shape=shape, low=low, high=high)
        self.observation_space = gym.spaces.Tuple(
            [obs_space for _ in range(self.n_agents)]
        )

        self.t = 0
        self.state = "A"

    def _make_obs(self):
        if self.state == "A":
            x = [1, 0, 0]
        elif self.state == "2A":
            x = [0, 1, 0]
        else:
            x = [0, 0, 1]
        return tuple([np.array(x)] * self.n_agents)

    def reset(self, seed=None, options=None):
        self.state = "A"
        self.t = 0
        # self.last_actions = actions_to_onehot(self.num_actions, [0] * self.n_agents)
        self.matrix1.reset()
        self.matrix2.reset()

        return self._make_obs(), {}

    def step(self, action):
        if self.t == 0:
            if action[0] == 0:
                self.state = "2A"
            else:
                self.state = "2B"
            rewards = self.n_agents * [0]
        elif self.t == 1:
            if self.state == "2A":
                _, rewards, _, _ = self.matrix1.step(action)
            elif self.state == "2B":
                _, rewards, _, _ = self.matrix2.step(action)
        else:
            rewards = self.n_agents * [0]

        done = self.t != 0
        truncated = false
        self.t += 1

        return self._make_obs(), rewards, done, truncated, {}


# penalty game
def create_penalty_game(penalty, ep_length, last_action_state=True):
    assert penalty <= 0
    payoff = [
        [penalty, 0, 10],
        [0, 2, 0],
        [10, 0, penalty],
    ]
    game = MatrixGame(payoff, ep_length, last_action_state)
    return game


# climbing game
def create_climbing_game(ep_length, last_action_state=True):
    payoff = [
        [11, -30, 0],
        [-30, 7, 0],
        [0, 6, 5],
    ]
    game = MatrixGame(payoff, ep_length, last_action_state)
    return game

