#%% Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

#%% Approximation function
def approximation_function(state):    
    dists_walls = np.array(state[0], dtype=np.float32).flatten() / 1000.0
    
    dists_cars = np.array(state[1], dtype=np.float32).flatten() / 300.0
    
    total_cps = len(state[3])
    if total_cps == 0: total_cps = 1.0 
    
    cp_index = np.array([state[2][0] / float(total_cps)], dtype=np.float32)
        
    speed = np.array([state[4]], dtype=np.float32).flatten()

    return np.concatenate([dists_walls, dists_cars, cp_index, speed])

#%% Standard DQN Model
class MyModel(nn.Module):
    def __init__(self, input_size=18, output_size=5): 
        super(MyModel, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

#%% MyAgent Class
class MyAgent:
    def __init__(self):
        self.best_reward = float('-inf')
        self.path = "records/race_model.pth" 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = MyModel(input_size=18, output_size=5).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) 
        self.criterion = nn.MSELoss() 

    def fit(self, X, y):
        self.model.train()
        features = np.array([approximation_function(x) for x in X], dtype=np.float32)
        targets = np.array(y, dtype=np.float32)
        
        features_tensor = torch.tensor(features).to(self.device)
        targets_tensor = torch.tensor(targets).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(features_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            features = approximation_function(state)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.model(features_tensor).cpu().numpy()[0]
    
    def predict_batch(self, states):
        self.model.eval()
        with torch.no_grad():
            features = np.array([approximation_function(s) for s in states], dtype=np.float32)
            features_tensor = torch.tensor(features).to(self.device)
            return self.model(features_tensor).cpu().numpy()
    
    def save(self, current_reward):
        print(f"\n    [SAVE] Aktualny wynik: {current_reward:.1f} (Rekord: {self.best_reward:.1f})")
        if current_reward >= self.best_reward:
            print(f"    >>> NOWY REKORD! Zapisuję model.")
            self.best_reward = current_reward
            save_data = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_reward": self.best_reward
            }
            os.makedirs('records', exist_ok=True)
            torch.save(save_data, self.path)

    def load(self):
        if os.path.exists(self.path):
            checkpoint = torch.load(self.path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.best_reward = checkpoint.get("best_reward", 0)
            print(f"    Załadowano model. Best Reward: {self.best_reward:.1f}.")
            return True, self.best_reward
        return False, None
    
    def smaller_learning_rate(self, factor=0.9):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor
        print(f"    Zmniejszono LR. Nowy: {self.optimizer.param_groups[0]['lr']:.6f}")