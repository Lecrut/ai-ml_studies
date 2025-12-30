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

        augmented_state = list(state) 
        augmented_state.append(self.vel / self.max_vel)
        
        action_index = int(self.agent.predict(augmented_state).argmax())
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

class PlayerCar2(AbstractCar):
    def __init__(self, name):
        super().__init__(name)
        car_agent = MyAgent()
        car_agent.load()
        self.agent = car_agent

    def choose_action(self, state):
        augmented_state = list(state) 
        augmented_state.append(self.vel / self.max_vel)
        
        action_index = int(self.agent.predict(augmented_state).argmax())
        actions = ["forward", "backward", "left", "right", "stop"]
        return actions[action_index]
    

def train_agent(num_episodes=5000,
                epsilon_start=1.0,
                epsilon_end=0.05,
                max_steps=2000, 
                gamma=0.99,
                batch_size=64,
                validate_every=20,
                epsilon_decay_steps=1000
                ):

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
                return random.choice(["forward", "forward", "left", "right", "stop"])
            else:
                q_values = self.agent.predict(state)
                actions = ["forward", "backward", "left", "right", "stop"]
                return actions[int(np.argmax(q_values))]

    def calculate_spawn_data(track_checkpoints, specific_index=None):
        if specific_index is not None:
            idx = specific_index
        else:
            idx = random.randint(0, len(track_checkpoints) - 5)
        
        curr_x, curr_y = track_checkpoints[idx]
        next_x, next_y = track_checkpoints[idx + 1]
        
        dx = next_x - curr_x
        dy = next_y - curr_y
        rads = math.atan2(dy, dx)
        angle = math.degrees(rads)
        
        adjusted_angle = 270 - angle 
        
        return curr_x, curr_y, adjusted_angle, idx

    agent = MyAgent()
    is_loaded = agent.load()
    if is_loaded:
        print("Model wczytany. Kontynuuję trening.")
        current_epsilon = epsilon_end 
    else:
        print("Nowy trening.")
        current_epsilon = epsilon_start

    replay_buffer = deque(maxlen=50000)
    game = Game(WIDTH, HEIGHT, FPS)
    
    best_validation_reward = float('-inf')

    try:
        iterator = trange(num_episodes, desc="Epizod")
    except NameError:
        iterator = range(num_episodes)

    actions_list = ["forward", "backward", "left", "right", "stop"]

    for episode in iterator:
        game.cars = []
        cars = []
        start_idx = None

        is_validation_run = (episode > 0) and (episode % validate_every == 0)
        
        if is_validation_run:
            current_epsilon = 0.0 
            start_idx = 0 
        elif is_loaded and episode == 0:
            current_epsilon = epsilon_end
        else:
            progress = (episode % epsilon_decay_steps) / epsilon_decay_steps
            progress = min(1.0, progress)
            current_epsilon = epsilon_end + (epsilon_start - epsilon_end) * ((1 - progress) ** 2)


        for i in range(4):
            eps = current_epsilon
            
            car = TrainingCar(f"AI_{i}", agent, min(1.0, eps))
            game.add_car(car)
            
            sx, sy, s_angle, s_idx = calculate_spawn_data(CHECKPOINTS, specific_index=start_idx)
            
            car.x, car.y = sx, sy
            car.angle = s_angle
            car.checkpoint_index = s_idx
            car.last_checkpoint = s_idx
            
            cars.append(car)

        states = []
        for car in cars:
            _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
            car_distances = car.get_distances_to_cars(game.cars)
            state = [distances, car_distances, car.get_progress(), CHECKPOINTS, car.vel / car.max_vel]
            states.append(state)

        step = 0
        done = [False] * len(cars)

        while not all(done) and step < max_steps:
            if step % 100 == 0: pygame.event.pump() 

            for i, car in enumerate(cars):
                if done[i]: continue

                state = states[i]
                action_str = car.choose_action(state)
                car.perform_action(action_str)
                car.update_progress(CHECKPOINTS)

                _, next_distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
                next_car_distances = car.get_distances_to_cars(game.cars)
                next_state = [next_distances, next_car_distances, car.get_progress(), CHECKPOINTS, car.vel / car.max_vel]

                reward = -0.1 
                norm_vel = car.vel / car.max_vel
                
                reward += norm_vel * 0.5

                if car.checkpoint_index > car.last_checkpoint:
                    reward += 50.0 
                    car.last_checkpoint = car.checkpoint_index
                    car.stuck_counter = 0
                    
                    if car.checkpoint_index >= len(CHECKPOINTS) - 2:
                        reward += 200

                if car.collide(TRACK_BORDER_MASK):
                    reward -= 50
                    done[i] = True
                
                if norm_vel < 0.05:
                    car.stuck_counter += 1
                else:
                    car.stuck_counter = 0
                
                if car.stuck_counter > 60: 
                    reward -= 20
                    done[i] = True

                action_idx = actions_list.index(action_str)
                replay_buffer.append((state, action_idx, reward, next_state, done[i]))
                states[i] = next_state
                car.total_reward += reward

            step += 1
            
            if len(replay_buffer) > batch_size and step % 4 == 0:
                batch = random.sample(replay_buffer, batch_size)
                
                states_b = [x[0] for x in batch]
                actions_b = [x[1] for x in batch]
                rewards_b = [x[2] for x in batch]
                next_states_b = [x[3] for x in batch]
                dones_b = [x[4] for x in batch]

                current_qs = [agent.predict(s) for s in states_b]
                next_qs = [agent.predict(s) for s in next_states_b]

                X = []
                y = []

                for k in range(batch_size):
                    current_q = current_qs[k].copy() 
                    target = rewards_b[k]
                    if not dones_b[k]:
                        target += gamma * np.max(next_qs[k])
                    
                    current_q[actions_b[k]] = target
                    X.append(states_b[k])
                    y.append(current_q)
                
                agent.fit(X, y)
        
        if is_validation_run:
            max_episode_reward = max(car.total_reward for car in cars)
            
            if iterator: 
                iterator.set_description(f"Valid: Rew {max_episode_reward:.1f} Best {best_validation_reward:.1f} Eps {current_epsilon:.2f}")

            if max_episode_reward > best_validation_reward:
                best_validation_reward = max_episode_reward
                agent.save(best_validation_reward)
                print(f" >>> ZAPISANO NOWY MODEL! (Reward: {best_validation_reward:.1f})")

    agent.save(best_validation_reward)
    print("Trening zakończony.")


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