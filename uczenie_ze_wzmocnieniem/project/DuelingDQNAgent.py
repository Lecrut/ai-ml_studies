import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DuelingQNetwork(nn.Module):
    def __init__(self, num_features, num_actions):
        super(DuelingQNetwork, self).__init__()

        # Wspólna warstwa wejściowa
        self.feature_layer = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Ścieżka Value (V) - jak dobry jest dany stan
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Ścieżka Advantage (A) - jak dobra jest dana akcja względem innych
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Łączenie V i A zgodnie ze wzorem Dueling DQN
        # Q(s,a) = V(s) + (A(s,a) - średnia(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DuelingDQNAgent:
    def __init__(self, num_features, num_actions, lr=0.0005, gamma=0.98, epsilon=1.0, epsilon_decay=0.9998, epsilon_min = 0.01):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay  # Wolniejszy decay niż w liniowym

        # Double DQN: Mamy dwie sieci - lokalną i docelową (target)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingQNetwork(num_features, num_actions).to(self.device)
        self.target_net = DuelingQNetwork(num_features, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=20000)  # Replay Buffer
        self.batch_size = 512

    def choose_action(self, state_features, eval_mode=False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        state_t = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return torch.argmax(q_values).item()

    def store_transition(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0

        # Losowanie paczki danych z pamięci
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 1. Wybór akcji przez sieć lokalną (Double DQN logic)
        curr_q = self.policy_net(states).gather(1, actions).squeeze(1)

        # 2. Ewaluacja akcji przez sieć Target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        return self