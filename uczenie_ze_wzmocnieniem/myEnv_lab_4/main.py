#%% Imports
from agents.QLearningAgent import QLearningAgent
from labyrinthMAP import LabyrinthMap
from tqdm import trange

#%% Training Function
def training(num_agents):
    env = LabyrinthMap(num_agents)

    max_tests = 400
    n_eps = 300
    eps = 0.1
    lr = 0.1

    for _ in trange(max_tests):
        eps += 0.002
        agents = [QLearningAgent(alpha=lr, epsilon=eps, discount=1,
                           get_legal_actions=env.get_possible_actions) for _ in range(num_agents)]
        for _ in range(n_eps):
            env.reset()
            done = [False for _ in range(num_agents)]
            while not any(done):
                for i, agent in enumerate(agents):
                    if done[i]:
                        continue
                    state = env._agents[i].get_current_state()
                    action = agent.get_action(state)
                    next_state, reward, terminal = env.step(i, action)
                    agent.update(state, action, reward, next_state)
                    if terminal:
                        done[i] = True

    print("Training completed.")

#%% Show Game 
def show_game(num_agents=2):
    env = LabyrinthMap(num_agents)
    env.reset()
    done = False
    agents = [QLearningAgent(alpha=0, epsilon=0, discount=1,
                       get_legal_actions=env.get_possible_actions) for _ in range(num_agents)]

    while not done:
        for i, agent in enumerate(agents):
            state = env._agents[i].get_current_state()
            action = agent.get_action(state)
            next_state, reward, terminal = env.step(i, action)
            agent.update(state, action, reward, next_state)

            env.show_map()
            if terminal:
                done = True
                break

#%% Run Game
if __name__ == "__main__":
    training(num_agents=2)

    show_game(num_agents=2)

