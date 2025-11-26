#%% Imports 
from labyrinthMAP import LabyrinthMap

#%% Show game
def show_game(env, is_show_map=False):
    pass

#%% Monte Carlo Tree Search

#%% Play game
if __name__ == "__main__":
    env = LabyrinthMap(2)

    states = env.get_all_states()
    # print('States:', states)

    # print_states_and_transitions(env, states)


    show_game(env, is_show_map=True)