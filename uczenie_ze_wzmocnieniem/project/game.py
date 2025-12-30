import pygame
from abstract_car import AbstractCar
from utils import scale_image
from itertools import permutations
import numpy as np
import math
from myAgent import MyAgent
import random
from collections import deque
from tqdm import trange
import time

#Based on https://github.com/techwithtim/Pygame-Car-Racer

GRASS = scale_image(pygame.image.load("project/imgs/grass.jpg"), 2.5)
TRACK = scale_image(pygame.image.load("project/imgs/track.png"), 0.9)

TRACK_BORDER = scale_image(pygame.image.load("project/imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("project/imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("project/imgs/red-car.png"), 0.35)
GREEN_CAR = scale_image(pygame.image.load("project/imgs/green-car.png"), 0.35)
PURPLE_CAR = scale_image(pygame.image.load("project/imgs/purple-car.png"), 0.35)
GRAY_CAR = scale_image(pygame.image.load("project/imgs/grey-car.png"), 0.35)


WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

pygame.font.init()  # Initialize the font module
FONT = pygame.font.Font(None, 24)  # Use a default font with size 24


FPS = 60

track_path =  [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]


# Interpolate evenly spaced checkpoints
def generate_checkpoints(track_path, num_checkpoints=100):
    checkpoints = []
    for i in range(len(track_path) - 1):
        x1, y1 = track_path[i]
        x2, y2 = track_path[i + 1]
        for t in np.linspace(0, 1, num_checkpoints // len(track_path)):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            checkpoints.append((int(x), int(y)))
    return checkpoints


CHECKPOINTS = generate_checkpoints(track_path)

def draw_checkpoints(win, checkpoints):
    for x, y in checkpoints:
        pygame.draw.circle(win, (0, 255, 0), (x, y), 5)

# In the game loop


class Game:
    def __init__(self, width, height, fps=60):
        self.win = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Racing Game")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.cars = []  # List to hold car objects
        self.images = [(GRASS, (0, 0)), (TRACK, (0, 0)),
          (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
        self.running = True

    def add_car(self, car):
        """Add a car to the game."""
        if not isinstance(car, AbstractCar):
            raise ValueError("Only instances of AbstractCar or its subclasses can be added.")

        if len(self.cars) == 0:
            car.set_image(RED_CAR)
            car.set_position((180, 200))
        elif len(self.cars) == 1:
            car.set_image(GREEN_CAR)
            car.set_position((150, 200))
        if len(self.cars) == 2:
            car.set_image(GRAY_CAR)
            car.set_position((180, 160))
        elif len(self.cars) == 3:
            car.set_image(PURPLE_CAR)
            car.set_position((150, 160))

        car.reset()
        self.cars.append(car)

    def draw(self):
        """Draw the background and all cars."""
        for img, pos in self.images:
            self.win.blit(img, pos)

        for car in self.cars:
            car.draw(self.win)
            car.draw_rays(self.win, TRACK_BORDER_MASK)


        pygame.display.update()

    def check_collisions(self):

        for car in self.cars:
            if car.collide(TRACK_BORDER_MASK):
                car.bounce()

        """Check for collisions between cars."""
        for i, car1 in enumerate(self.cars):
            for j, car2 in enumerate(self.cars):
                if i != j and car1.collide_car(car2):
                    car1.bounce()
                    car2.bounce()
                    # print(f"Collision between Car {i+1} and Car {j+1}!")

    def check_finish_line(self):

        finished = []

        for car in self.cars:
            finish_poi_collide = car.collide(FINISH_MASK, *FINISH_POSITION)
            if finish_poi_collide != None:
                if finish_poi_collide[1] == 0:
                    car.bounce()
                else:
                    finished.append(car.get_name())
                    self.cars.remove(car)

        return finished

    def move_cars(self):
        """Handle car movements."""

        for car in self.cars:
            car.update_progress(CHECKPOINTS)

        for car in self.cars:
            _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
            car_distances = car.get_distances_to_cars(self.cars)
            car.perform_action(car.choose_action([distances, car_distances, car.get_progress(), CHECKPOINTS]))
    def run(self):
        """Main game loop."""
        who_finished_first = []
        while self.running and len(self.cars) != 0:
            self.clock.tick(self.fps)
            draw_checkpoints(self.win, CHECKPOINTS)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False



            self.move_cars()
            self.check_collisions()
            finish_lines = self.check_finish_line()
            if len(finish_lines) != 0:
                who_finished_first.append(finish_lines)

            self.draw()


        pygame.quit()
        print("Game over!")
        print(who_finished_first)
        return who_finished_first


class PlayerCar(AbstractCar):

    def __init__(self, name):
        # Call the AbstractCar __init__ method
        super().__init__(name)

    def choose_action(self, state):
        """
        Perform an action based on the input.

        Actions:
        - "forward": Move the car forward.
        - "backward": Move the car backward.
        - "left": Turn the car left.
        - "right": Turn the car right.
        - "stop": Reduce the car's speed.
        """

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            return "forward"
        elif keys[pygame.K_DOWN]:
            return "backward"
        elif keys[pygame.K_LEFT]:
            return "left"
        elif keys[pygame.K_RIGHT]:
            return "right"
        else:
            return "stop"


class PlayerCar2(AbstractCar):

    def __init__(self, name):
        super().__init__(name)

        car_agent = MyAgent()
        car_agent.load()
        self.agent = car_agent

    def choose_action(self, state):
        """
        Determines the next action for the car based on the current state of the environment.

        Parameters:
            state (list): A 3-element list representing the car's current state:
                - state[0]: A list of 8 float values representing distances to the track border
                            in 8 directions (every 45 degrees, starting from forward).
                - state[1]: A list of 8 float values representing distances to the nearest car
                           in the same 8 directions.
                - state[2]: A 2-element list representing progress information:
                            - state[2][0]: The index of the closest checkpoint.
                            - state[2][1]: The car's progress, e.g., distance to the next checkpoint
                                           or normalized progress value.

        Returns:
            - "forward": Move the car forward.
            - "backward": Move the car backward.
            - "left": Turn the car left.
            - "right": Turn the car right.
            - "stop": Reduce the car's speed.
            """

        action_index = int(self.agent.predict(state).argmax())
        actions = ["forward", "backward", "left", "right", "stop"]
        return actions[action_index]

        # keys = pygame.key.get_pressed()

        # if keys[pygame.K_w]:
        #     return "forward"
        # elif keys[pygame.K_s]:
        #     return "backward"
        # elif keys[pygame.K_a]:
        #     return "left"
        # elif keys[pygame.K_d]:
        #     return "right"
        # else:
        #     return "stop"

def train_agent(num_episodes=10000,
                epsilon_start=1.0,
                epsilon_end=0.01,
                max_steps=5000,
                gamma=0.99,
                batch_size=64):
    
    class TrainingCar(AbstractCar):
        def __init__(self, name, agent_model, epsilon_val):
            super().__init__(name)
            self.agent = agent_model
            self.epsilon = epsilon_val
            self.last_checkpoint = 0
            self.total_reward = 0
            self.stuck_counter = 0
            
        def choose_action(self, state):
            if random.random() < self.epsilon:
                return random.choice(["forward", "forward", "forward", "left", "right", "backward"])
            else:
                q_values = self.agent.predict(state)
                actions = ["forward", "backward", "left", "right", "stop"]
                return actions[int(np.argmax(q_values))]

    agent = MyAgent()
    if not agent.load():
        print("Nowy trening...")
    
    replay_buffer = deque(maxlen=100000) 
    epsilon = epsilon_start
 
    exploration_episodes = int(num_episodes * 0.8)
    epsilon_decay_value = (epsilon_start - epsilon_end) / exploration_episodes
    
    actions_list = ["forward", "backward", "left", "right", "stop"]
    
    game = Game(WIDTH, HEIGHT, FPS)
    
    best_reward = float('-inf')
    
    try:
        iterator = trange(num_episodes, desc="Epizod")
    except NameError:
        iterator = range(num_episodes)

    for _ in iterator:
        game.cars = []
        cars = []
        for i in range(4):
            car = TrainingCar(f"AI_Train_{i+1}", agent, epsilon)
            game.add_car(car)
            cars.append(car)
        
        states = []
        prev_data = [] 
        
        for car in cars:
            _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
            car_distances = car.get_distances_to_cars(game.cars)
            state = [distances, car_distances, car.get_progress(), CHECKPOINTS]
            states.append(state)
            
            cur_check_x, cur_check_y = CHECKPOINTS[car.checkpoint_index]
            prev_dist_to_cp = math.sqrt((car.x - cur_check_x)**2 + (car.y - cur_check_y)**2)
            prev_pos = (car.x, car.y)
            prev_data.append((prev_dist_to_cp, prev_pos))
        
        step = 0
        done = [False] * 4 

        while not all(done) and step < max_steps:
            if step % 100 == 0: pygame.event.pump()
            
            for car_idx, car in enumerate(cars):
                if done[car_idx]:
                    continue
                    
                state = states[car_idx]
                prev_dist_to_cp, prev_pos = prev_data[car_idx]
                
                action_str = car.choose_action(state)
                action_idx = actions_list.index(action_str)
                
                car.perform_action(action_str)
                car.update_progress(CHECKPOINTS)
                
                _, next_distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
                next_car_distances = car.get_distances_to_cars(game.cars)
                next_state = [next_distances, next_car_distances, car.get_progress(), CHECKPOINTS]
                
                reward = 0
                
                if action_str == "forward":
                    reward += 1.5 * (car.vel / car.max_vel)
                elif action_str == "stop":
                    reward -= 0.1 
                
                cur_check_x, cur_check_y = CHECKPOINTS[car.checkpoint_index]
                curr_dist_to_cp = math.sqrt((car.x - cur_check_x)**2 + (car.y - cur_check_y)**2)
                delta = prev_dist_to_cp - curr_dist_to_cp
                
                if delta > 0:
                    reward += delta * 4.0 
                else:
                    reward += delta * 6.0 
                
                if car.checkpoint_index > car.last_checkpoint:
                    reward += 200 
                    car.last_checkpoint = car.checkpoint_index
                    car.stuck_counter = 0 
                    curr_dist_to_cp = math.sqrt((car.x - CHECKPOINTS[car.checkpoint_index][0])**2 + (car.y - CHECKPOINTS[car.checkpoint_index][1])**2)
                
                if car.collide(TRACK_BORDER_MASK):
                    reward -= 150 
                    done[car_idx] = True
                
                dist_moved = math.sqrt((car.x - prev_pos[0])**2 + (car.y - prev_pos[1])**2)
                if dist_moved < 2: 
                    car.stuck_counter += 1
                else:
                    car.stuck_counter = 0
                    
                if car.stuck_counter > 50:
                    reward -= 50
                    done[car_idx] = True

                finish_collide = car.collide(FINISH_MASK, *FINISH_POSITION)
                if finish_collide is not None and finish_collide[1] != 0:
                    reward += 1000 
                    done[car_idx] = True

                replay_buffer.append((state, action_idx, reward, next_state, done[car_idx]))
                
                states[car_idx] = next_state
                prev_data[car_idx] = (curr_dist_to_cp, (car.x, car.y))
                car.total_reward += reward
            
            step += 1
            
            if len(replay_buffer) > batch_size and step % 5 == 0: 
                batch = random.sample(replay_buffer, batch_size)
                
                states_batch = [x[0] for x in batch]
                next_states_batch = [x[3] for x in batch]
                
                current_qs_list = [agent.predict(s) for s in states_batch]
                next_qs_list = [agent.predict(s) for s in next_states_batch]
                
                X_train = []
                y_train = []
                
                for i in range(batch_size):
                    _, action, r, _, d = batch[i]
                    current_q = current_qs_list[i].copy()
                    if d:
                        current_q[action] = r
                    else:
                        current_q[action] = r + gamma * np.max(next_qs_list[i])
                    X_train.append(states_batch[i])
                    y_train.append(current_q)
                
                agent.fit(X_train, y_train)

        if epsilon > epsilon_end:
            epsilon -= epsilon_decay_value
            epsilon = max(epsilon_end, epsilon)
        
        total_episode_reward = sum(car.total_reward for car in cars)
        
        if total_episode_reward > best_reward:
            best_reward = total_episode_reward
            agent.save(total_episode_reward)
        
      
    agent.save(best_reward)
    print(f"\nTrening zakończony! Najlepszy wynik: {best_reward:.2f}")
def main():
    final_results = dict()

    #initializing players - it is possible to play up to 4 players together
    # players = [PlayerCar("P1"), PlayerCar2("P2"), PlayerCar("P1"), PlayerCar2("P2")]
    players = [PlayerCar2("AI_1"), PlayerCar2("AI_2"), PlayerCar2("AI_3"), PlayerCar2("AI_4")]

    for p in players:
        final_results[p.get_name()] = 0

    perm = permutations(players)

    for p in perm:

        print(p)

        game = Game(WIDTH, HEIGHT, FPS)

        # Add cars
        for player in p:
            game.add_car(player)

        # Run the game
        temp_rank = game.run()

        points = len(players)

        for tr in temp_rank:
            for t in tr:
                final_results[t] += points
            points -= 1

    print(final_results)

if __name__ == "__main__":
    train_agent()

    main()