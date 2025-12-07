#%% Imports 
from labyrinthMAP import LabyrinthMap, get_goal_position, MAP, WIDTH, HEIGHT, LEFT, DOWN, RIGHT, UP
from sklearn import neural_network
from tqdm import trange
import random

#%% Approximation function 
def approximation_function(state):
    me = state[0]
    mx, my = me.get_position()
    gx, gy = get_goal_position()

    others_pos = {ag.get_position() for ag in state[1:]}

    dist_x = (gx - mx) / WIDTH
    dist_y = (gy - my) / HEIGHT

    sensors = []
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for dx, dy in moves:
        tx, ty = mx + dx, my + dy
        is_blocked = 0
        
        if not (0 <= tx < WIDTH and 0 <= ty < HEIGHT) or MAP[ty][tx] == 'W':
            is_blocked = 1

        elif (tx, ty) in others_pos:
            is_blocked = 1
            
        sensors.append(is_blocked)

    return [dist_x, dist_y] + sensors

#%% Teacher Logic
def get_teacher_move(state):
    me = state[0]
    mx, my = me.get_position()
    gx, gy = get_goal_position()
    others_pos = {ag.get_position() for ag in state[1:]}

    valid_moves = []
    moves = [(LEFT, -1, 0), (DOWN, 0, 1), (RIGHT, 1, 0), (UP, 0, -1)]

    for action, dx, dy in moves:
        nx, ny = mx + dx, my + dy
        
        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
            if MAP[ny][nx] != 'W' and (nx, ny) not in others_pos:
                dist = abs(gx - nx) + abs(gy - ny)
                valid_moves.append((dist, action))
    
    if not valid_moves:
        return random.choice([LEFT, DOWN, RIGHT, UP])
    
    valid_moves.sort(key=lambda x: x[0])
    return valid_moves[0][1]

#%% MLP Agent
class MyAgent:
    def __init__(self):
        self.model = neural_network.MLPClassifier(
            hidden_layer_sizes=(30, 20), 
            max_iter=500,
            activation='relu',
            learning_rate_init=1e-2
        )

    def fit(self, X, y):
        features = [approximation_function(x) for x in X]
        self.model.fit(features, y)

    def predict(self, x):          
        features = approximation_function(x)
        return self.model.predict([features])[0]

#%% Train Agent
def train_agent(agent, env, episodes=5000):
    X = []
    y = []

    for _ in trange(episodes):
        env.reset()
        done = [False for _ in range(env._num_agents)]

        while not any(done): 
            for i in range(env._num_agents):
                current_state = env._agents[i].get_current_state()

                if env.is_terminal(current_state):
                    done[i] = True
                    break

                correct_action = get_teacher_move(current_state)

                X.append(current_state)
                y.append(correct_action)

                _, _, terminal = env.step(i, correct_action)

                if terminal:
                    done[i] = True

    agent.fit(X, y)

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
            _, reward, terminal = env.step(i, action)
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
    train_agent(agent, env, episodes=100)

    show_game(env, agent, show_map=True)