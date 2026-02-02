import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class NeuralModel(nn.Module):
    def __init__(self, input_size=18, output_size=5): 
        super(NeuralModel, self).__init__()
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

class MyAgentPiotr:
    def __init__(self):
        self.best_reward = -9999
        self.path = "records/best_model.pth" 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = NeuralModel(input_size=18, output_size=5).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) 
        self.criterion = nn.MSELoss() 
        
        self.load()

    def evaluate_state(self, state):
        walls = np.array(state[0], dtype=np.float32).flatten() / 1000.0
        cars = np.array(state[1], dtype=np.float32).flatten() / 300.0
        
        checkpoints_len = len(state[3]) if isinstance(state[3], list) else 1.0
        if checkpoints_len == 0: checkpoints_len = 1.0
        checkpoint_progress = np.array([state[2][0] / float(checkpoints_len)], dtype=np.float32)
        
        velocity = np.array([state[4]], dtype=np.float32).flatten() / 10.0

        return np.concatenate([walls, cars, checkpoint_progress, velocity])


    def fit(self, X, y):
        self.model.train()
        
        X_np = np.array([self.evaluate_state(x) for x in X], dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)
        X_tensor = torch.tensor(X_np).to(self.device)
        y_tensor = torch.tensor(y_np).to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            features = self.evaluate_state(state)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.model(features_tensor).cpu().numpy()[0]

    def predict_batch(self, states):
        self.model.eval()
        with torch.no_grad():
            features = np.array([self.evaluate_state(s) for s in states], dtype=np.float32)
            features_tensor = torch.tensor(features).to(self.device)
            return self.model(features_tensor).cpu().numpy()
    
    def save(self, current_reward):
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            save_data = {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_reward": self.best_reward
            }
            if not os.path.exists('records'):
                os.makedirs('records')
            torch.save(save_data, self.path)
            print(f"Nowy rekord: {self.best_reward:.2f}")

    def load(self):
        if os.path.exists(self.path):
            try:
                checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)
                
                self.model.load_state_dict(checkpoint["model_state"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.best_reward = checkpoint.get("best_reward", float('-inf'))
                print(f"Załadowano model. Rekord: {self.best_reward:.2f}")
            except Exception as e:
                print(f"Błąd ładowania: {e}")
                print("Startuję od zera.")