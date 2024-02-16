from random import sample
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# To organize files and keep track of information, not nesseccary, but usefull
from model_organizer.organizer import HANDLER


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
        self.constants = constants
        self.policy_net = DQN(constants.N_ACTIONS).to(constants.DEVICE)

        self.handler = HANDLER(
            "./models",
            load_model,
            self.policy_net,
        )

        self.handler.adjust_required_info(
            [
                "model name",
                "batch size",
                "episodes",
                "learning rate",
                "complete",
                "top score",
                "top SAS",
            ]
        )

        if load_model:
            self.policy_net = self.handler.load_ai_model(load_model)
            self.policy_net.eval()

        self.target_net = DQN(constants.N_ACTIONS).to(constants.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(
            self.policy_net.parameters(), lr=constants.LR, amsgrad=True
        )
        self.memory = ReplayMemory(10000)

    def store_state_action(self, state, action, next_state, reward):
        # Create reward tensor
        reward_tensor = torch.tensor([reward], device=self.constants.DEVICE)

        # Create a transition
        self.memory.push(state, action, next_state, reward_tensor)

    def soft_update_target_net(self):
        # Soft update of the target network's weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.constants.TAU + target_net_state_dict[key] * (
                1 - self.constants.TAU
            )
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):
        if len(self.memory) < self.constants.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.constants.BATCH_SIZE)

        # Transpose the batch
        batch = self.memory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.constants.DEVICE,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Get the actions from current policy net from stored state
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Get actions with the highest Q-value from the target net
        next_state_values = torch.zeros(
            self.constants.BATCH_SIZE, device=self.constants.DEVICE
        )
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.constants.GAMMA
        ) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Return the loss for display
        return round(loss.item(), 2)
