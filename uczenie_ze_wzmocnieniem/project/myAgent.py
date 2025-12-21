#%% Imports
from sklearn.neural_network import MLPRegressor
import numpy as np
import pickle
import os

#%% Approximation function
def approximation_function(state):  
    distances = np.array(state[0], dtype=np.float32)  
    car_distances = np.array(state[1], dtype=np.float32)  
    progress = np.array(state[2], dtype=np.float32) 
    
    max_distance = 1000.0
    distances_norm = distances / max_distance
    car_distances_norm = car_distances / max_distance
    
    progress_norm = progress.copy()
    progress_norm[0] = progress[0] / 100.0 
    
    
    avg_wall_distance = np.mean(distances_norm)
    min_wall_distance = np.min(distances_norm)
    avg_car_distance = np.mean(car_distances_norm)
    min_car_distance = np.min(car_distances_norm)
    
    features = np.concatenate([
        distances_norm,      
        car_distances_norm,   
        progress_norm,       
        [avg_wall_distance, min_wall_distance, avg_car_distance, min_car_distance] 
    ])
    
    return features

#%% Approximation-based Agent
class MyAgent:
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=1,  
            warm_start=True,
            random_state=42
        )
        
        self.model.fit(
            [np.zeros(22)], 
            [np.zeros(4)]
            )
        
        self.path = f"records/race.pkl"

    def fit(self, X, y):
        features = [approximation_function(x) for x in X]
        self.model.partial_fit(features, y)

    def predict(self, x):          
        features = approximation_function(x)
        return self.model.predict([features])[0]
    
    def save(self):
        os.makedirs('records', exist_ok=True)
        try:
            with open(self.path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"\nAgent saved to: {self.path}")
        except Exception as e:
            print(f"\nSave error: {e}")

    def load(self):
        try:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"\nAgent loaded from: {self.path}")
            return True
        except FileNotFoundError:
            print(f"\nModel file {self.path} not found. Starting new training.")
            return False
        except Exception as e:
            print(f"\nModel loading error: {e}. Starting new training.")
            return False
