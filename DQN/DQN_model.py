import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.ReplayMemory import ReplayMemory


class DQN_Model(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        load_model: Optional[str] = None,
        memory_size: int = 10000,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 1000,
        model_class: nn.Module = DQN_Model,
    ):
        self.memory = ReplayMemory(capacity=memory_size)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.out_dim = output_dim
        if load_model is None:
            self.policy_net: nn.Module = model_class(input_dim, output_dim)
            self.target_net: nn.Module = model_class(input_dim, output_dim)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
        else:
            self.policy_net = torch.load(load_model, weights_only=False)
            self.target_net = torch.load(load_model, weights_only=False)
            self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.steps_done = 0

    def optimize_model(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
            *batch
        )
        state_batch = torch.FloatTensor(np.array(state_batch))
        action_batch = torch.LongTensor(np.array(action_batch)).unsqueeze(1)
        reward_batch = torch.FloatTensor(np.array(reward_batch))
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        done_batch = torch.FloatTensor(np.array(done_batch))
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        with torch.no_grad():
            max_next_q_value = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_value * (
                1 - done_batch
            )

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state: np.ndarray, training: bool = False) -> int:
        if training and random.random() < self.epsilon:
            return random.randint(0, self.out_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy_net(state)
        self.steps_done += 1
        return int(torch.argmax(q_values).item())

    def update_model(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(state, action, reward, next_state, done)
        self.optimize_model()
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
