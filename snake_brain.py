from random import sample
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple(
            "Transition", ("state", "action", "next_state", "reward")
        )

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(9, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x


class SNAKE_BRAIN:
    def __init__(self, load_model, constants):
        if load_model:
            self.policy_net = DQN(constants.N_ACTIONS).to(constants.DEVICE)
            # TODO: Don't use try except (use handler for this anyway)
            try:
                self.policy_net.load_state_dict(
                    torch.load(f"./models/model_{load_model}")
                )
            except:
                self.policy_net.load_state_dict(
                    torch.load(f"./models/model_{load_model}_incomplete")
                )
            self.policy_net.eval()
        else:
            self.policy_net = DQN(constants.N_ACTIONS).to(constants.DEVICE)

        self.target_net = DQN(constants.N_ACTIONS).to(constants.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=constants.LR, amsgrad=True
        )
        self.memory = ReplayMemory(10000)
