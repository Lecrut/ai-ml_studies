from agents.QLearningAgent import QLearningAgent
from labyrinthMAP import LabyrinthMap


def training(num_agents):
    env = LabyrinthMap(num_agents)

    max_tests = 500
    n_eps = 300
    eps = 0.1
    lr = 0.1

    t = 0
    while t < max_tests:
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
        t += 1
        print(f"Completed {t}/{max_tests} training iterations.")

    print("Training completed.")

if __name__ == "__main__":
    training(num_agents=2)
