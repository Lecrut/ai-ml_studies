import pygame
from abstract_car import AbstractCar
from utils import scale_image
from itertools import permutations
import numpy as np
import math
from myAgent import MyAgent
import random
import copy
from collections import deque
from tqdm import trange, tqdm

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
    def __init__(self, width, height, fps=60, render=True):
        self.render = render

        if self.render:
            self.win = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Racing Game")
            self.clock = pygame.time.Clock()
        else:
            self.win = None
            self.clock = None

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
        else :
            car.set_image(RED_CAR)
            car.set_position((180, 200))

        car.reset()
        self.cars.append(car)

    def draw(self):
        """Draw the background and all cars."""
        if not self.render:
            return
    
        for img, pos in self.images:
            self.win.blit(img, pos)

        for car in self.cars:
            car.draw(self.win)
            #car.draw_rays(self.win, TRACK_BORDER_MASK)


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
            if self.render and self.clock:
                self.clock.tick(self.fps)
                # draw_checkpoints(self.win, CHECKPOINTS)
                pygame.display.update()

            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False



            self.move_cars()
            self.check_collisions()
            finish_lines = self.check_finish_line()
            if len(finish_lines) != 0:
                who_finished_first.append(finish_lines)

            self.draw()

        if self.render:
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
    

def train_agent(num_episodes=2000,
                epsilon_start=1.0,
                batch_size=256,        
                epsilon_decay_steps=1000,
                frame_skip=2
                ):

    # --- KONFIGURACJA ---
    RWD_WALL = -2.0           
    RWD_IDLE = -5.0          
    RWD_CHECKPOINT = 40.0     
    RWD_FINISH = 200.0        
    RWD_SPEED = 2.0           
    
    CARS_TRAINING = 20  
    CARS_VALIDATION = 4 

    class TrainingCar(AbstractCar):
        def __init__(self, name, agent_model, epsilon_val):
            super().__init__(name)
            self.agent = agent_model
            self.epsilon = epsilon_val
            self.last_checkpoint = 0
            self.total_reward = 0
            self.frames_stuck = 0
            self.current_action_idx = 0 
            self.cached_state = None 

        def choose_action_idx(self, state):
            if random.random() < self.epsilon:
                return random.randint(0, 4)
            return int(np.argmax(self.agent.predict(state)))
        
        def choose_action(self, state):
            actions = ["forward", "backward", "left", "right", "stop"]
            return actions[self.current_action_idx]
        
        def get_state(self, game_cars, track_mask, checkpoints):
            _, distances = self.get_rays_and_distances(track_mask)
            car_distances = self.get_distances_to_cars(game_cars)
            return [distances, car_distances, self.get_progress(), checkpoints, self.vel / self.max_vel]

    agent = MyAgent()
    is_loaded, _ = agent.load()
    target_agent = copy.deepcopy(agent)
    
    replay_buffer = deque(maxlen=40000)
    gamma = 0.95 
    
    iterator = trange(num_episodes, desc="Trening")
    
    for episode in iterator:
        if is_loaded:
            epsilon = max(0.05, epsilon_start * (0.995 ** episode))
        else:
            ratio = min(1.0, episode / epsilon_decay_steps)
            epsilon = epsilon_start * (1.0 - ratio) + 0.05 * ratio

        is_validation = (episode % 20 == 0) and (episode > 0)
        current_num_cars = CARS_VALIDATION if is_validation else CARS_TRAINING
        
        game = Game(WIDTH, HEIGHT, fps=60, render=is_validation)
        
        cars = []
        
        for i in range(current_num_cars): 
            eps = 0.0 if is_validation else epsilon
            car = TrainingCar(f"AI_{i}", agent, eps)
            game.add_car(car)
            
            if is_validation:
                car.x, car.y = (170 + i * 10, 200 - i * 10) 
                car.angle = 0
                car.checkpoint_index = 0
                car.last_checkpoint = 0
            else:
                spawn_idx = random.randint(0, len(CHECKPOINTS) - 3)
                pos = CHECKPOINTS[spawn_idx]
                next_pos = CHECKPOINTS[spawn_idx+1]
                dx, dy = next_pos[0] - pos[0], next_pos[1] - pos[1]
                angle = math.degrees(math.atan2(dy, dx))
                
                car.x = pos[0] + random.randint(-10, 10)
                car.y = pos[1] + random.randint(-10, 10)
                car.angle = 270 - angle 
                car.checkpoint_index = spawn_idx
                car.last_checkpoint = spawn_idx
            
            car.cached_state = car.get_state(game.cars, TRACK_BORDER_MASK, CHECKPOINTS)
            cars.append(car)

        step = 0
        active_cars = True
        max_steps = 2000 if is_validation else 600
        
        while active_cars and step < max_steps: 
            if game.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); return

            for car in cars:
                if car not in game.cars: continue
                if step % frame_skip == 0:
                    action_idx = car.choose_action_idx(car.cached_state)
                    car.current_action_idx = action_idx
            
            game.move_cars()
            game.check_collisions()
            game.check_finish_line()

            if game.render: game.draw()

            if step % frame_skip == 0:
                for car in cars:
                    if car not in game.cars: continue

                    next_state = car.get_state(game.cars, TRACK_BORDER_MASK, CHECKPOINTS)
                    reward = 0
                    done = False
                    
                    if car.checkpoint_index > car.last_checkpoint:
                        reward += RWD_CHECKPOINT
                        car.last_checkpoint = car.checkpoint_index
                        car.frames_stuck = 0
                        if car.checkpoint_index >= len(CHECKPOINTS) - 2:
                            reward += RWD_FINISH
                            done = True
                    else:
                        car.frames_stuck += frame_skip
                        
                    norm_vel = car.vel / car.max_vel
                    if norm_vel > 0.1: reward += norm_vel * RWD_SPEED
                    else: reward += RWD_IDLE

                    if car.collide(TRACK_BORDER_MASK):
                        reward += RWD_WALL
                        done = True

                    if car.frames_stuck > 50:
                        reward += RWD_IDLE
                        done = True

                    car.total_reward += reward
                    
                    if not is_validation:
                        replay_buffer.append((car.cached_state, car.current_action_idx, reward, next_state, done))
                    
                    car.cached_state = next_state
                    
                    if done:
                        game.cars.remove(car)

            if not is_validation and len(replay_buffer) > batch_size and step % 5 == 0:
                batch = random.sample(replay_buffer, batch_size)
                states = np.array([x[0] for x in batch], dtype=object)
                actions = np.array([x[1] for x in batch])
                rewards = np.array([x[2] for x in batch], dtype=np.float32)
                next_states = np.array([x[3] for x in batch], dtype=object)
                dones = np.array([x[4] for x in batch], dtype=np.float32)

                current_qs = agent.predict_batch(states)
                next_qs = target_agent.predict_batch(next_states)
                max_next_qs = np.max(next_qs, axis=1)
                target_qs = rewards + (1 - dones) * gamma * max_next_qs
                
                target_vals = current_qs.copy()
                rows = np.arange(batch_size)
                target_vals[rows, actions] = target_qs
                
                agent.fit(states, target_vals)

            if not is_validation and step % 100 == 0:
                target_agent = copy.deepcopy(agent)
            
            step += 1
            if len(game.cars) == 0: active_cars = False

        if len(cars) > 0:
            max_reward = max([c.total_reward for c in cars])
            mode_str = "WALIDACJA" if is_validation else "Trening"
            iterator.set_description(f"[{mode_str}] Best: {max_reward:.0f} | Eps: {epsilon:.2f}")
            
            if is_validation and max_reward > agent.best_reward:
                agent.save(max_reward)

    agent.save(agent.best_reward)
    pygame.quit()


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