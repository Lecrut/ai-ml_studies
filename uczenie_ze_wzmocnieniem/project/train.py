import pygame
import numpy as np
import math
import random
import copy
from collections import deque
from tqdm import tqdm

from abstract_car import AbstractCar
from myAgent import MyAgent
from utils import scale_image

pygame.init()

try:
    GRASS = scale_image(pygame.image.load("project/imgs/grass.jpg"), 2.5)
    TRACK = scale_image(pygame.image.load("project/imgs/track.png"), 0.9)
    TRACK_BORDER = scale_image(pygame.image.load("project/imgs/track-border.png"), 0.9)
    TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)
    
    RED_CAR = scale_image(pygame.image.load("project/imgs/red-car.png"), 0.35)
    GREEN_CAR = scale_image(pygame.image.load("project/imgs/green-car.png"), 0.35)
    GRAY_CAR = scale_image(pygame.image.load("project/imgs/grey-car.png"), 0.35)
    PURPLE_CAR = scale_image(pygame.image.load("project/imgs/purple-car.png"), 0.35)
    CAR_IMAGES = [RED_CAR, GREEN_CAR, GRAY_CAR, PURPLE_CAR]
except Exception:
    TRACK_BORDER_MASK = pygame.mask.Mask((100, 100))
    CAR_IMAGES = [None] * 4

track_path = [(175, 119), (110, 70), (56, 133), (70, 481), (318, 731), (404, 680), (418, 521), (507, 475), (600, 551), (613, 715), (736, 713),
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

class HeadlessCar(AbstractCar):
    def __init__(self, name, agent_model, epsilon_val, img):
        super().__init__(name)
        self.agent = agent_model
        self.epsilon = epsilon_val
        self.img = img
        if self.img: self.mask = pygame.mask.from_surface(self.img) 
        else: self.mask = pygame.mask.Mask((10,10)) 

        self.last_checkpoint = 0
        self.total_reward = 0
        self.frames_since_checkpoint = 0 
        self.prev_total_progress = 0.0
        self.current_action_idx = 0 
        self.cached_state = None 
        self.done = False

    def choose_action_idx(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        return int(np.argmax(self.agent.predict(state)))

    def get_state(self, game_cars, track_mask, checkpoints):
        _, distances = self.get_rays_and_distances(track_mask)
        car_distances = self.get_distances_to_cars(game_cars)
        return [distances, car_distances, self.get_progress(), checkpoints, self.vel / self.max_vel]
    
    def get_exact_progress(self, checkpoints):
        cp_idx = self.checkpoint_index
        next_cp_idx = (cp_idx + 1) % len(checkpoints)
        cur_cp_pos = checkpoints[cp_idx]
        next_cp_pos = checkpoints[next_cp_idx]
        
        segment_len = math.hypot(next_cp_pos[0] - cur_cp_pos[0], next_cp_pos[1] - cur_cp_pos[1])
        if segment_len == 0: segment_len = 1.0
        
        dist_to_next = math.hypot(next_cp_pos[0] - self.x, next_cp_pos[1] - self.y)
        ratio = 1.0 - (dist_to_next / segment_len)
        return float(cp_idx) + max(0.0, min(1.0, ratio))

    def draw(self, win): pass

def train_headless(num_episodes=3000):
    BATCH_SIZE = 512
    GAMMA = 0.99            
    FRAME_SKIP = 2
    TAU = 0.005             

    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    
    EPSILON_DECAY_FRAC = 0.75
    decay_steps = int(num_episodes * EPSILON_DECAY_FRAC)

    CHECKPOINT_TIMEOUT = 200

    NUM_CARS_TRAIN = 15
    NUM_CARS_VALID = 4
    VALIDATION_EVERY = 50
    
    FINE_TUNING_PCT = 0.1

    START_POSITIONS = [(180, 200), (150, 200), (180, 160), (150, 160)]

    RWD_WALL = -1.0
    RWD_TIMEOUT = -1.0      
    RWD_CHECKPOINT = 5.0   
    RWD_FINISH = 50.0       
    RWD_SPEED = 0.02        
    RWD_DELTA_PROGRESS = 5.0 
    RWD_WRONG_WAY = -0.5    
    RWD_STEER_IDLE = -0.05   

    agent = MyAgent()
    is_loaded, _ = agent.load()
    target_agent = copy.deepcopy(agent)
    replay_buffer = deque(maxlen=100_000)
    actions_list = ["forward", "backward", "left", "right", "stop"]
    best_validation_reward = agent.best_reward

    pbar = tqdm(range(num_episodes), desc="Training")

    for episode in pbar:
        is_finetuning = episode >= (num_episodes * (1.0 - FINE_TUNING_PCT))
        is_validation = (episode % VALIDATION_EVERY == 0 and episode > 0)

        if is_validation:
            epsilon = 0.0
        else:
            progress = min(1.0, episode / decay_steps)
            epsilon = EPSILON_START - (progress * (EPSILON_START - EPSILON_MIN))

        cars = []
        
        if is_validation or is_finetuning:
            num_cars = NUM_CARS_VALID
        else:
            num_cars = NUM_CARS_TRAIN

        for i in range(num_cars):
            img = CAR_IMAGES[i % len(CAR_IMAGES)]
            car = HeadlessCar(f"AI_{i}", agent, epsilon, img)
            car.frames_since_checkpoint = 0
            
            if is_validation or is_finetuning:
                car.x, car.y = START_POSITIONS[i % len(START_POSITIONS)]
                car.angle = 0
                car.checkpoint_index = 0
                car.last_checkpoint = 0
                car.prev_total_progress = 0.0 
            else:
                safe_max_idx = len(CHECKPOINTS) - 2
                curriculum_steps = int(num_episodes * 0.5)
                progress_ratio = min(1.0, episode / curriculum_steps)
                
                max_spawn = int(safe_max_idx * progress_ratio)
                spawn_limit = min(max(5, max_spawn), safe_max_idx)
                
                spawn_idx = random.randint(0, spawn_limit)
                
                pos = CHECKPOINTS[spawn_idx]
                nxt = CHECKPOINTS[spawn_idx + 1]
                dx, dy = nxt[0] - pos[0], nxt[1] - pos[1]
                angle_deg = math.degrees(math.atan2(dy, dx))
                
                car.x = pos[0] + random.randint(-15, 15)
                car.y = pos[1] + random.randint(-15, 15)
                car.angle = 270 - angle_deg + random.randint(-15, 15)
                car.checkpoint_index = spawn_idx
                car.last_checkpoint = spawn_idx
                car.prev_total_progress = car.get_exact_progress(CHECKPOINTS)

            cars.append(car)

        for car in cars:
            car.cached_state = car.get_state(cars, TRACK_BORDER_MASK, CHECKPOINTS)

        step = 0
        hard_limit = 3000 if is_validation else 1500

        while step < hard_limit and any(not c.done for c in cars):
            for car in cars:
                if car.done: continue
                if step % FRAME_SKIP == 0:
                    car.current_action_idx = car.choose_action_idx(car.cached_state)
                car.perform_action(actions_list[car.current_action_idx])
                car.update_progress(CHECKPOINTS)

            if step % FRAME_SKIP == 0:
                for car in cars:
                    if car.done: continue
                    next_state = car.get_state(cars, TRACK_BORDER_MASK, CHECKPOINTS)
                    reward = 0.0

                    curr_total_progress = car.get_exact_progress(CHECKPOINTS)
                    delta = curr_total_progress - car.prev_total_progress
                    
                    if delta > 0:
                        reward += delta * RWD_DELTA_PROGRESS
                        if delta > 0.01: 
                            car.frames_since_checkpoint = 0
                    
                    car.prev_total_progress = curr_total_progress

                    if car.checkpoint_index > car.last_checkpoint:
                        reward += RWD_CHECKPOINT
                        car.last_checkpoint = car.checkpoint_index
                        if car.checkpoint_index >= len(CHECKPOINTS) - 2:
                            reward += RWD_FINISH
                            car.done = True
                    else:
                        vel_norm = car.vel / car.max_vel
                        penalty_factor = 0.3 + 0.7 * (1.0 - abs(vel_norm))
                        car.frames_since_checkpoint += FRAME_SKIP * penalty_factor

                    if car.frames_since_checkpoint > CHECKPOINT_TIMEOUT:
                        reward += RWD_TIMEOUT
                        car.done = True

                    if not car.done:
                        next_cp_idx = (car.checkpoint_index + 1) % len(CHECKPOINTS)
                        next_cp = CHECKPOINTS[next_cp_idx]
                        dx = next_cp[0] - car.x
                        dy = next_cp[1] - car.y
                        target_angle = math.atan2(dy, dx)
                        car_angle = math.radians(-car.angle)
                        angle_diff = abs(math.atan2(math.sin(target_angle - car_angle), math.cos(target_angle - car_angle)))
                        reward += RWD_WRONG_WAY * (angle_diff / math.pi)

                    vel_norm = car.vel / car.max_vel
                    reward += vel_norm * RWD_SPEED 
                    
                    if actions_list[car.current_action_idx] in ["left", "right"] and abs(vel_norm) < 0.2:
                        reward += RWD_STEER_IDLE
                        
                    if car.collide(TRACK_BORDER_MASK):
                        reward += RWD_WALL
                        car.done = True

                    car.total_reward += reward

                    if not is_validation:
                        is_significant = abs(reward) > 0.05 or car.done
                        if is_significant or random.random() < 0.05:
                             replay_buffer.append((car.cached_state, car.current_action_idx, reward, next_state, car.done))
                             
                    car.cached_state = next_state

            if not is_validation and len(replay_buffer) > BATCH_SIZE and step % 10 == 0:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                states = np.array([b[0] for b in batch], dtype=object)
                actions = np.array([b[1] for b in batch])
                rewards = np.array([b[2] for b in batch], dtype=np.float32)
                next_states = np.array([b[3] for b in batch], dtype=object)
                dones = np.array([b[4] for b in batch], dtype=np.float32)

                next_actions = np.argmax(agent.predict_batch(next_states), axis=1)
                next_qs = target_agent.predict_batch(next_states)
                target_q = rewards + (1 - dones) * GAMMA * next_qs[np.arange(BATCH_SIZE), next_actions]

                current_qs = agent.predict_batch(states)
                target_vals = current_qs.copy()
                target_vals[np.arange(BATCH_SIZE), actions] = target_q

                agent.fit(states, target_vals)

                for tp, lp in zip(target_agent.model.parameters(), agent.model.parameters()):
                    tp.data.copy_(TAU * lp.data + (1 - TAU) * tp.data)
            step += 1

        best = max(c.total_reward for c in cars) if cars else 0
        
        if is_validation: status = "VAL"
        elif is_finetuning: status = "FINE"
        else: status = "TRN"
            
        pbar.set_description(f"[{status}] Best: {best:.2f} | eps: {epsilon:.2f}")

        if is_validation:
            if best > best_validation_reward:
                best_validation_reward = best
                agent.save(best)
        elif episode % 50 == 0:
            agent.save(best)

    agent.save(agent.best_reward)
    print("TRENING ZAKOŃCZONY")

if __name__ == "__main__":
    train_headless()