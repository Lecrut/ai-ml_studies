#%% Imports
from sklearn.neural_network import MLPRegressor
import numpy as np
import pickle
import os

#%% Funkcja przetwarzająca dane wejściowe (Feature Engineering)
def approximation_function(
        state=None,
        max_ray_distance=1000.0,
        estimated_max_dist=10000.0,
        max_checkpoints=50.0
        ):  
    distances = np.array(state[0], dtype=np.float32)   
    car_distances = np.array(state[1], dtype=np.float32)  
    progress = np.array(state[2], dtype=np.float32)       
    
    distances_norm = distances / max_ray_distance
    car_distances_norm = car_distances / max_ray_distance
    
    progress_norm = np.zeros_like(progress)
    progress_norm[0] = progress[0] / max_checkpoints
    progress_norm[1] = progress[1] / estimated_max_dist 
    
    avg_wall_dist = np.mean(distances_norm)
    min_wall_dist = np.min(distances_norm)

    left_side = np.mean(distances_norm[0:4])
    right_side = np.mean(distances_norm[4:8])
    side_bias = left_side - right_side 

    features = np.concatenate([
        distances_norm,       
        car_distances_norm,   
        progress_norm,        
        [avg_wall_dist, min_wall_dist, side_bias]
    ])
    
    return features

#%% Główna klasa Agenta
class MyAgent:
    def __init__(self):
        self.best_reward = float('-inf')
        self.path = "records/race.pkl"
        
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
            [np.zeros(21)], 
            [np.zeros(5)] 
        )

    def fit(self, X, y):
        features = [approximation_function(x) for x in X]
        self.model.partial_fit(features, y)

    def predict(self, x):          
        features = approximation_function(x)
        return self.model.predict([features])[0] 
    
    def save(self, current_reward):
        if current_reward > self.best_reward:
            print(f"\n  Zapis - Nowy rekord! ({current_reward:.2f} > {self.best_reward:.2f}). Zapisywanie...")
            self.best_reward = current_reward
            
            save_data = {
                "model": self.model,
                "best_reward": self.best_reward
            }
            
            os.makedirs('records', exist_ok=True)
            try:
                with open(self.path, 'wb') as f:
                    pickle.dump(save_data, f)
            except Exception as e:
                print(f"Błąd zapisu pliku: {e}")

    def load(self):
        try:
            with open(self.path, 'rb') as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.best_reward = data["best_reward"]
                
                print(f"\nZaładowano Agenta. Rekord do pobicia: {self.best_reward:.2f}")
                return True
                
        except FileNotFoundError:
            print(f"\nBrak zapisu '{self.path}'. Tworzę nowego agenta.")
            return False
        except Exception as e:
            print(f"\nBłąd ładowania (plik może być uszkodzony): {e}. Tworzę nowego agenta.")
            return False