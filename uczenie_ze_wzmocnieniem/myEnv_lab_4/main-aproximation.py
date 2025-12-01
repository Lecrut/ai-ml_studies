#%% Imports
from labyrinthMAP import LabyrinthMap, WIDTH, HEIGHT, MAP
import numpy as np
import random
from tqdm import tqdm, trange

#%% Feature extraction
def extract_features(env, state):
    x, y = state[0].get_position()
    gx, gy = env._goal_state.get_position()

    max_d = WIDTH + HEIGHT - 2
    dist = abs(x - gx) + abs(y - gy)
    dist_norm = dist / max_d

    def is_wall(pos):
        x, y = pos
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        return MAP[y][x] == 'W'

    wu = 1 if is_wall((x-1, y)) else 0
    wd = 1 if is_wall((x+1, y)) else 0
    wl = 1 if is_wall((x, y-1)) else 0
    wr = 1 if is_wall((x, y+1)) else 0

    return np.array([dist_norm, wu, wd, wl, wr, 1.0], dtype=float)


#%% Linear value function approximation
class ValueFunctionApprox:
    def __init__(self, feature_dim=6, alpha=0.05, gamma=0.9):
        self.w = np.zeros(feature_dim)
        self.alpha = alpha
        self.gamma = gamma

    def V(self, phi):
        return np.dot(self.w, phi)

    def update(self, phi_s, reward, phi_next):
        target = reward + self.gamma * self.V(phi_next)
        prediction = self.V(phi_s)
        delta = target - prediction
        self.w += self.alpha * delta * phi_s


#%% Passive TD Training
def passive_td_training(env, episodes=2000):
    VF = ValueFunctionApprox()

    for _ in trange(episodes):
        env.reset()
        done = False
        agent = env._agents[0]     

        while not done:
            s = agent.get_current_state()
            phi_s = extract_features(env, s)

            actions = env.get_only_possible_actions(s)
            a = random.choice(actions) 

            _, r, done = env.step(0, a)

            s2 = agent.get_current_state()
            phi_next = extract_features(env, s2)

            VF.update(phi_s, r, phi_next)

    return VF

#%% Make greedy policy from approximated V(s)
def make_greedy_policy(env, VF):
    def policy(state):
        best_a = None
        best_val = -1e9

        actions = env.get_only_possible_actions(state)
        for a in tqdm(actions, desc="Building greedy policy"):
            next_states = env.get_next_states(state, a)
            if len(next_states) == 0:
                continue

            next_s = next_states[0]
            phi_next = extract_features(env, next_s)
            val = VF.V(phi_next)

            if val > best_val:
                best_val = val
                best_a = a

        return best_a

    return policy


#%% Show game
def show_game(env, policy, is_show_map=False):
    env.reset()
    done = [False for _ in range(env._num_agents)]
    rewards = [0 for _ in range(env._num_agents)]
    steps = 0

    while not any(done) and steps < 10000:
        steps += 1
        for i, agent in enumerate(env._agents):
            if done[i]:
                continue

            current_state = agent.get_current_state()
            action = policy(current_state)

            if not is_show_map:
                print(f"Agent {i} at state {current_state} takes action {action}")

            _, reward, terminal = env.step(i, action)
            rewards[i] += reward

            if is_show_map:
                env.show_map(number_of_moves=steps)

            if terminal:
                done[i] = True
                print(f"Agent {i} finished with reward: {rewards[i]} in {steps} steps")
                break

    if not any(done):
        print('None agent ended in less than 10000 steps.')

#%% Play game
if __name__ == "__main__":
    env = LabyrinthMap(2)

    print("Training passive TD approximator...")
    VF = passive_td_training(env, episodes=5000)

    print("Building greedy policy from approximated V(s)...")
    policy = make_greedy_policy(env, VF)

    print("Running game...")
    show_game(env, policy, is_show_map=True)
