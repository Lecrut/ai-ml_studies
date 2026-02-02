import random
from collections import deque
import numpy as np
import torch
from torch import nn
import math
from abc import abstractmethod
from abstract_car import AbstractCar



class CarMovementNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.layers(x)


class DQNMemory:
    def __init__(self, capacity, batch_size):
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.memory.append((state, action, next_state, reward))

    def sample(self):
        if 5*self.batch_size > len(self.memory):
            return None
        sample = random.sample(self.memory, self.batch_size)
        return map(list, zip(*sample)) # noqa

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, input_size, output_size, lr=1e-3, gamma=0.99, batch_size=256, memory_capacity=50000, epsilon=0.8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy_net = CarMovementNN(input_size, output_size).to(self.device)
        self.target_net = CarMovementNN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = DQNMemory(memory_capacity, batch_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.learning = True

    def get_best_action(self, state):
        dqn_values = self.get_values(state).view(-1)
        dqn_values_cpu = dqn_values.cpu().numpy()
        max_value = np.max(dqn_values_cpu)
        best_actions_list = np.argwhere(dqn_values_cpu == max_value).flatten()
        best_action = random.choice(best_actions_list).item()
        return best_action

    def get_action(self, state):
        epsilon = self.epsilon
        random_value = random.random()
        chosen_action = random.randint(0, 3) if random_value < epsilon else self.get_best_action(state)

        return chosen_action

    def get_values(self, state):
        with torch.no_grad():
            q = self.policy_net(torch.from_numpy(state).float().to(self.device))
        return q

    def train_step(self):
        if self.learning:
            buffer = self.buffer.sample()
            if buffer is None:
                return
            s, a, ns, r = buffer

            s = torch.tensor(np.array(s), device=self.device).float()
            ns = torch.tensor(np.array(ns), device=self.device).float()
            a = torch.tensor(np.array(a), device=self.device)
            r = torch.tensor(np.array(r), device=self.device).float()

            q = self.policy_net(s)
            q = q.gather(1, a.unsqueeze(1))
            q = q.squeeze(1)

            with torch.no_grad():
                # next_actions = self.policy_net(ns).max(1)[1].unsqueeze(1)
                # qn = self.target_net(ns).gather(1, next_actions).squeeze(1)
                qn = self.target_net(ns).max(1)[0]
                target = r + self.gamma * qn

            criterion = nn.MSELoss()
            loss = criterion(q, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            "model": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["model"])
        self.target_net.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def test(self):
        self.epsilon = 0
        self.learning = False

    def train(self):
        self.learning = True

TEST = True

class AgentCar(AbstractCar):
    def __init__(self, name, state_representation_size, outputs, load_filename=None):
        super().__init__(name)
        self.step = 0
        self.state_representation_size = state_representation_size
        self.epsilon_start = 0.1
        self.epsilon_end = 1e-5
        self.epsilon = self.epsilon_start
        self.agent = DQNAgent(state_representation_size, outputs, batch_size=128, epsilon=self.epsilon_start)
        self.agent.load(self.name if load_filename is None else load_filename)
        self.previous_state = None
        self.previous_action = None
        self.previous_velocity = 0
        

    def get_reward(self, distance):
        progress_threshold = 0.25
        if self.checkpoint_index != self.previous_state[1][0]:
            progress_reward = 10
            # if TEST:
            #     print(f'{self.name}({self.state_representation_size}) new cp {self.checkpoint_index}!')
            time_reward = -0.01
        else:
            track_progress = self.previous_state[1][1] - distance
            progress_reward = 1.5*(track_progress-progress_threshold) if track_progress > progress_threshold \
                else (track_progress-progress_threshold)*3
            time_reward = -0.5 if abs(track_progress) < 0.5 else -0.01
        crash_reward = -15 if self.crashed == 1 and progress_reward < 0 else 0
        crash_reward += -25 if self.crashed == 2 else 0

        return progress_reward + time_reward + crash_reward

    @abstractmethod
    def prepare_state_representation(self, state):
        pass

    def choose_action(self, state):
        state_representation = self.prepare_state_representation(state)
        if self.checkpoint_index < len(state[3]):
            checkpoint_x, checkpoint_y = state[3][self.checkpoint_index]
            distance = math.sqrt((self.x - checkpoint_x) ** 2 + (self.y - checkpoint_y) ** 2)
        else:
            checkpoint_x, checkpoint_y = state[3][0]
            distance = math.sqrt((self.x - checkpoint_x) ** 2 + (self.y - checkpoint_y) ** 2)

        # if self.previous_state is not None:
        #     reward = self.get_reward(distance)
        #     self.crashed = 0
        #     if not TEST:
        #         self.agent.buffer.push(self.previous_state[0], self.previous_action,
        #                           state_representation, reward)

        self.previous_state = (state_representation, (self.checkpoint_index, distance))
        # if self.step%(math.ceil(self.epsilon*5)) == 0 or not self.agent.learning:
        self.previous_action = self.agent.get_action(self.previous_state[0])
        if self.agent.learning:
            self.agent.epsilon = self.epsilon

        self.agent.train_step()
        self.step += 1

        if self.previous_action == 0:
            return "forward"
        elif self.previous_action == 1:
            return "backward"
        elif self.previous_action == 2:
            return "left"
        elif self.previous_action == 3:
            return "right"

        return "forward"


class PlayerCarSensorsProgressMinimumSensorsValuesAndVelocity(AgentCar):
    def __init__(self, name, outputs, load_filename=None):
        super().__init__(name, 21, outputs, load_filename)
        self.previous_cp = -1
        self.start_cp_dist = 1000

    def prepare_state_representation(self, state):
        # Wall distances
        state_representation = state[0]
        state_representation.append(min(state[0]))
        # Cars distances
        state_representation.extend(state[1])
        state_representation.append(min(state[1]))

        if self.checkpoint_index < len(state[3]):
            checkpoint_x, checkpoint_y = state[3][self.checkpoint_index]
            distance = math.sqrt((self.x - checkpoint_x) ** 2 + (self.y - checkpoint_y) ** 2)
        else:
            checkpoint_x, checkpoint_y = state[3][0]
            distance = math.sqrt((self.x - checkpoint_x) ** 2 + (self.y - checkpoint_y) ** 2)

        if self.checkpoint_index != self.previous_cp:
            self.previous_cp = self.checkpoint_index
            self.start_cp_dist = distance

        # Progress
        state_representation.append(self.checkpoint_index / len(state[3]))
        state_representation.append(((distance - 40) / (self.start_cp_dist - 40)))

        # Velocity
        state_representation.append(np.clip(self.vel, -8, 8))

        # Normalization
        state_representation = np.array(state_representation, dtype=np.float32)
        state_representation[:18] = np.clip(state_representation[:18], 7, 100) - 7
        state_representation[:18] /= 93
        state_representation[20] /= 8

        return state_representation
