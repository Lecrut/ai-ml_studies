#%% Imports
from agents.QLearningAgent import QLearningAgent
from labyrinthMAP import LabyrinthMap
from tqdm import trange
import matplotlib.pyplot as plt

#%% Training Function
def training(num_agents):
    env = LabyrinthMap(num_agents)

    max_tests = 400
    n_eps = 300
    eps = 0.9
    lr = 0.1
    discount = 0.9
    best_agents_rewards = []

    for _ in trange(max_tests):
        eps -= 0.002
        agents = [QLearningAgent(alpha=lr, epsilon=eps, discount=discount,
                           get_legal_actions=env.get_possible_actions) for _ in range(num_agents)]
        rewards = [0 for _ in range(num_agents)]
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
                    rewards[i] += reward
                    if terminal:
                        done[i] = True
                        best_agents_rewards.append(rewards[i])
                        break

    print("Training completed.")

    plt.plot(best_agents_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards over Episodes during Training')
    plt.show()

#%% Show Game 
def show_game(num_agents=2):
    env = LabyrinthMap(num_agents)
    env.reset()
    done = False
    agents = [QLearningAgent(alpha=0, epsilon=0, discount=1,
                       get_legal_actions=env.get_possible_actions) for _ in range(num_agents)]

    for agent in agents:
        agent.turn_off_learning()

    while not done:
        for i, agent in enumerate(agents):
            state = env._agents[i].get_current_state()
            action = agent.get_action(state)
            _, _, terminal = env.step(i, action)

            env.show_map()
            if terminal:
                done = True
                break

#%% Run Game
if __name__ == "__main__":
    training(num_agents=2)

    show_game(num_agents=2)

