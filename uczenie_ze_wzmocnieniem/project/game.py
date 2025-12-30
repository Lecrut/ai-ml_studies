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


def train_agent(num_episodes=1000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.998, 
                max_steps=3000, gamma=0.99, batch_size=64):
    
    agent = MyAgent()
    agent.load()  
    
    replay_buffer = deque(maxlen=20000)
    epsilon = epsilon_start
    
    actions = ["forward", "backward", "left", "right", "stop"]
    
    for episode in trange(num_episodes):
        
        class TrainingCar(AbstractCar):
            def __init__(self, name, agent_model, epsilon_val):
                super().__init__(name)
                self.agent = agent_model
                self.epsilon = epsilon_val
                self.last_checkpoint = 0
                self.total_reward = 0
                self.collided = False
                self.last_position = None
                self.stuck_counter = 0
                self.last_distance_to_checkpoint = None
                
            def choose_action(self, state):
                if random.random() < self.epsilon:
                    exploration_actions = ["forward", "forward", "left", "right", "left", "right"]
                    return random.choice(exploration_actions)
                else:
                    q_values = self.agent.predict(state)
                    return actions[int(q_values.argmax())]
            
            def get_angle_to_checkpoint(self, checkpoints):
                if self.checkpoint_index >= len(checkpoints):
                    return 0.0
                
                checkpoint_x, checkpoint_y = checkpoints[self.checkpoint_index]
                
                dx = checkpoint_x - (self.x + self.img.get_width() // 2)
                dy = checkpoint_y - (self.y + self.img.get_height() // 2)
                
                angle_to_checkpoint = math.degrees(math.atan2(-dy, dx))
                
                relative_angle = angle_to_checkpoint - self.angle
                
                while relative_angle > 180:
                    relative_angle -= 360
                while relative_angle < -180:
                    relative_angle += 360
                
                return relative_angle
        
        car = TrainingCar("Training_AI", agent, epsilon)
        
        game = Game(WIDTH, HEIGHT, FPS)
        game.add_car(car)
        
        step = 0
        episode_experiences = []
        done = False
        last_checkpoint_step = 0
        
        while not done and step < max_steps:
            _, distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
            car_distances = car.get_distances_to_cars(game.cars)
            prev_checkpoint = car.checkpoint_index
            angle_to_checkpoint = car.get_angle_to_checkpoint(CHECKPOINTS)
            state = [distances, car_distances, car.get_progress(), [angle_to_checkpoint]]
            
            action_str = car.choose_action(state)
            action_idx = actions.index(action_str)
            
            current_pos = (car.x, car.y)
            if car.last_position is not None:
                distance_moved = np.sqrt((car.x - car.last_position[0])**2 + (car.y - car.last_position[1])**2)
                if distance_moved < 1:
                    car.stuck_counter += 1
                else:
                    car.stuck_counter = 0
            car.last_position = current_pos
            
            car.perform_action(action_str)
            
            reward = 0
            
            if action_str == "backward":
                reward -= 2.0  
            
            if action_str == "forward" and car.vel > 0:
                reward += 1.0  
                reward += 0.5 * (car.vel / car.max_vel)  
            
            if abs(angle_to_checkpoint) > 30:  
                if action_str == "left" and angle_to_checkpoint > 0:
                    reward += 1.5  
                elif action_str == "right" and angle_to_checkpoint < 0:
                    reward += 1.5  
            
            if car.checkpoint_index < len(CHECKPOINTS):
                checkpoint_x, checkpoint_y = CHECKPOINTS[car.checkpoint_index]
                current_distance = np.sqrt((car.x - checkpoint_x)**2 + (car.y - checkpoint_y)**2)
                
                if car.last_distance_to_checkpoint is not None:
                    distance_delta = car.last_distance_to_checkpoint - current_distance
                    if car.vel >= 0:
                        reward += distance_delta * 1.0  
                
                car.last_distance_to_checkpoint = current_distance
            
            new_angle_to_checkpoint = car.get_angle_to_checkpoint(CHECKPOINTS)
            if abs(new_angle_to_checkpoint) < abs(angle_to_checkpoint):
                reward += 1.0  
            
            car.update_progress(CHECKPOINTS)
            if car.checkpoint_index > car.last_checkpoint:
                checkpoint_reward = (car.checkpoint_index - car.last_checkpoint) * 300  # HUGE reward!
                reward += checkpoint_reward
                car.last_checkpoint = car.checkpoint_index
                last_checkpoint_step = step
                car.last_distance_to_checkpoint = None  

            if car.stuck_counter > 20:
                reward -= 10
            
            if step - last_checkpoint_step > 300:
                reward -= 5
            
            if car.collide(TRACK_BORDER_MASK):
                reward -= 20  
                car.collided = True
                car.bounce()
            
            finish_collide = car.collide(FINISH_MASK, *FINISH_POSITION)
            if finish_collide is not None and finish_collide[1] != 0:
                time_bonus = max(0, 1000 - step) 
                reward += 1000 + time_bonus
                done = True
            
            _, next_distances = car.get_rays_and_distances(TRACK_BORDER_MASK)
            next_car_distances = car.get_distances_to_cars(game.cars)
            next_angle_to_checkpoint = car.get_angle_to_checkpoint(CHECKPOINTS)
            next_state = [next_distances, next_car_distances, car.get_progress(), [next_angle_to_checkpoint]]
            
            episode_experiences.append((state, action_idx, reward, next_state, done))
            car.total_reward += reward
            
            step += 1
            
            if step > 400 and car.checkpoint_index == 0:
                reward -= 20  
                done = True
        
        replay_buffer.extend(episode_experiences)
        
        num_training_iterations = 3 if len(replay_buffer) >= batch_size else 1
        
        for _ in range(num_training_iterations):
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                
                X_train = []
                y_train = []
                
                for state, action, reward, next_state, is_done in batch:
                    current_q = agent.predict(state)
                    
                    if is_done:
                        target_q = reward
                    else:
                        next_q = agent.predict(next_state)
                        target_q = reward + gamma * np.max(next_q)
                    
                    target_q_values = current_q.copy()
                    target_q_values[action] = target_q
                    
                    X_train.append(state)
                    y_train.append(target_q_values)
                
                agent.fit(X_train, y_train)
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if (episode + 1) % 50 == 0:
            agent.save()
            print(f"    Model saved!")
        
    agent.save()

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