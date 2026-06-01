import math
from collections import deque
from itertools import permutations
from pathlib import Path

import numpy as np
import pygame
from tqdm import trange, tqdm

from abstract_car import AbstractCar
from agent_expert import MyAgent
from utils import scale_image

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
WIN = pygame.Surface((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

pygame.font.init()  # Initialize the font module
FONT = pygame.font.Font(None, 24)  # Use a default font with size 24


FPS = 60
FRAME_SKIP = 4
FRAME_STACK_SIZE = 4
CAPTURE_SIZE = (256, 256)
DATASET_BATCH_SIZE = 1000
DATASET_OUTPUT_DIR = Path("project/dataset")
NUM_EPISODES = 5
ACTION_NAMES = ["forward", "backward", "left", "right", "stop"]

track_path =  [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
        (734, 399), (611, 357), (409, 343), (433, 257), (697, 258), (738, 123), (581, 71), (303, 78), (275, 377), (176, 388), (178, 260)]


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


def preprocess_frame(surface, size=CAPTURE_SIZE):
    scaled_surface = pygame.transform.smoothscale(surface, size)
    rgb_frame = pygame.surfarray.array3d(scaled_surface).astype(np.uint8)
    return np.transpose(rgb_frame, (1, 0, 2))


def action_to_index(action):
    try:
        return ACTION_NAMES.index(action)
    except ValueError as error:
        raise ValueError(f"Unknown action: {action}") from error


class DatasetCollector:
    def __init__(self, output_dir=DATASET_OUTPUT_DIR, batch_size=DATASET_BATCH_SIZE, stack_size=FRAME_STACK_SIZE):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stack_size = stack_size
        self.frame_buffer = deque(maxlen=stack_size)
        self.sample_index = 1

    def add_frame(self, surface):
        self.frame_buffer.append(preprocess_frame(surface))

    def record_sample(self, action):
        if len(self.frame_buffer) < self.stack_size:
            return False
        stacked_frames = np.stack(self.frame_buffer, axis=0)
        action_idx = action_to_index(action)

        output_path = self.output_dir / f"sample_{self.sample_index:06d}.npz"
        np.savez_compressed(output_path, states=stacked_frames, actions=np.array(action_idx, dtype=np.uint8))
        self.sample_index += 1

        return True

    def flush(self):
        return None

    def close(self):
        return None



class Game:
    def __init__(self, width, height, fps=60, render=True):
        self.render = render

        if self.render:
            self.win = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Racing Game")
            self.clock = pygame.time.Clock()
        else:
            self.win = pygame.Surface((width, height))
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

        car.reset()
        car.current_action = "stop"
        self.cars.append(car)

    def draw(self):
        """Draw the background and all cars."""

        for img, pos in self.images:
            self.win.blit(img, pos)

        for car in self.cars:
            car.draw(self.win)
            #car.draw_rays(self.win, TRACK_BORDER_MASK)

        if self.render:
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

    def _get_car_state(self, car):
        _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
        car_distances = car.get_distances_to_cars(self.cars)
        return [distances, car_distances, car.get_progress(), CHECKPOINTS]

    def run_for_dataset(self, collector, frame_skip=FRAME_SKIP, max_steps=None):
        """Run the game loop while collecting imitation-learning samples."""
        who_finished_first = []
        step = 0

        while self.running and len(self.cars) != 0:
            if max_steps is not None and step >= max_steps:
                break

            if self.render and self.clock:
                self.clock.tick(self.fps)

            if self.render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

            decision_frame = (step % frame_skip == 0)

            for car in self.cars:
                car.update_progress(CHECKPOINTS)
                
            collector.add_frame(self.win)

            if decision_frame:
                self.draw()

                for car in self.cars:
                    action = car.choose_action(self._get_car_state(car))
                    car.current_action = action
                    collector.record_sample(action)

            for car in self.cars:
                car.perform_action(car.current_action)

            self.check_collisions()
            finish_lines = self.check_finish_line()
            if len(finish_lines) != 0:
                who_finished_first.append(finish_lines)

            if self.render:
                self.draw()

            step += 1

        collector.close()

        if self.render:
            pygame.quit()

        print("Game over!")
        print(who_finished_first)
        return who_finished_first

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


def main():
    collector = DatasetCollector(output_dir=DATASET_OUTPUT_DIR, batch_size=DATASET_BATCH_SIZE)

    for _ in range(NUM_EPISODES):
        game = Game(WIDTH, HEIGHT, FPS, render=False)
        expert = PlayerCar2("AI_1")
        game.add_car(expert)
        game.run_for_dataset(collector, frame_skip=FRAME_SKIP)

    collector.close()

if __name__ == "__main__":
    main()