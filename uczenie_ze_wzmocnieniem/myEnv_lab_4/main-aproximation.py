#%% Imports
from labyrinthMAP import LabyrinthMap, get_goal_position, MAP, WIDTH, HEIGHT, LEFT, DOWN, RIGHT, UP
from sklearn.neural_network import MLPRegressor
from tqdm import trange
import numpy as np
import pickle
import os

#%% Constants
AVAILABLE_ACTIONS = [LEFT, DOWN, RIGHT, UP]

#%% Feature extraction 
def get_radar(mx, my, others_pos):
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)] 
    readings = []
    
    max_range = 5.0

    for dx, dy in moves:
        dist = 0
        for i in range(1, int(max_range) + 1):
            tx, ty = mx + (dx * i), my + (dy * i)
            
            if not (0 <= tx < WIDTH and 0 <= ty < HEIGHT) or MAP[ty][tx] == 'W':
                break
            if (tx, ty) in others_pos:
                break
            dist += 1
            
        readings.append(dist / max_range)
        
    return readings

def approximation_function(state):
    me = state[0]
    mx, my = me.get_position()
    gx, gy = get_goal_position()
    
    others_pos = {ag.get_position() for ag in state[1:]}

    pos_x = mx / WIDTH
    pos_y = my / HEIGHT

    target_dx = (gx - mx) / WIDTH
    target_dy = (gy - my) / HEIGHT
    
    radar = get_radar(mx, my, others_pos)
    
    return np.array([pos_x, pos_y, target_dx, target_dy] + radar)

#%% Approximation-based Agent
class MyAgent:
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=1,  
            warm_start=True,
            random_state=42
        )
        
        self.model.fit(
            [np.zeros(8)], 
            [np.zeros(4)]
            )
        
        self.path = 'records/approximation.pkl'

    def fit(self, X, y):
        features = [approximation_function(x) for x in X]
        self.model.partial_fit(features, y)

    def predict(self, x):          
        features = approximation_function(x)
        return self.model.predict([features])[0]
    
    def save(self):
        os.makedirs('records', exist_ok=True)
        try:
            with open(self.path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"\nAgent saved to: {self.path}")
        except Exception as e:
            print(f"\nSave error: {e}")

    def load(self):
        try:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"\nAgent loaded from: {self.path}")
            return True
        except FileNotFoundError:
            print(f"\nModel file {self.path} not found. Starting new training.")
            return False
        except Exception as e:
            print(f"\nModel loading error: {e}. Starting new training.")
            return False

#%% Training loop
def train_agent(agent, env, episodes=5000, max_steps=500):
    gamma = 0.9 
    eps = 0.9

    for episode in trange(episodes):
        eps -= (eps / 2 + 0.1) / episodes
        env.reset()
        done = [False for _ in range(env._num_agents)]
        steps = 0

        X = []
        y = []
        
        while not any(done) and steps < max_steps: 
            steps += 1
            for i in range(env._num_agents):
                current_state = env._agents[i].get_current_state()

                if env.is_terminal(current_state):
                    done[i] = True
                    break

                current_q_values = agent.predict(current_state)
                
                if np.random.rand() < eps: 
                    action_idx = np.random.randint(0, 4)
                else:
                    action_idx = np.argmax(current_q_values) 

                taken_action = AVAILABLE_ACTIONS[action_idx]
                next_state, reward, terminal = env.step(i, taken_action) 
                target_q = current_q_values.copy()

                if terminal:
                    target_q[action_idx] = reward
                else:
                    next_q_values = agent.predict(next_state)
                    max_future_q = np.max(next_q_values)
                    target_q[action_idx] = reward + gamma * max_future_q

                X.append(current_state)
                y.append(target_q)

                if terminal:
                    done[i] = True
                    break

        if episode % 5 == 0 or episode == 0:
            agent.fit(X, y)
            X, y = [], []

#%% Show game
def show_game(env, agent, show_map=True):
    env.reset()
    steps = 0
    done = [False for _ in range(env._num_agents)]
    rewards = [0 for _ in range(env._num_agents)]

    while not any(done) and steps < 1000:
        steps += 1

        for i in range(env._num_agents):
            if env.is_terminal(env._agents[i].get_current_state()):
                continue

            action = agent.predict(env._agents[i].get_current_state())
            next_action = AVAILABLE_ACTIONS[np.argmax(action)]

            _, reward, terminal = env.step(i, next_action)
            rewards[i] += reward

            if show_map:
                env.show_map(number_of_moves=steps)
            
            if terminal:
                done[i] = True
                print(f"Agent {i} finished with reward: {rewards[i]} in {steps} steps")
                break

    if not any(done):
        print('None agent ended in less than 200 steps.')     

#%% Play game
if __name__ == "__main__":
    env = LabyrinthMap(2) 

    agent = MyAgent()
    agent.load()

    # train_agent(agent, env) 

    # agent.save()

    show_game(env, agent, show_map=True)