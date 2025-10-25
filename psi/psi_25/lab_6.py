import numpy as np


def objective_function(pos):
    x, y = pos
    term1 = (1.5 - x - x * y) ** 2
    term2 = (2.25 - x + x * y ** 2) ** 2
    term3 = (2.625 - x + x * y ** 3) ** 2
    return term1 + term2 + term3


def initialize_particles(num_particles, bounds):
    positions = np.random.uniform(bounds[0], bounds[1], (num_particles, 2))
    velocities = np.random.uniform(-1, 1, (num_particles, 2))
    return positions, velocities


def update_velocity(velocity, position, pbest, gbest, w, c1, c2):
    r1 = np.random.rand(*position.shape)
    r2 = np.random.rand(*position.shape)
    cognitive = c1 * r1 * (pbest - position)
    social = c2 * r2 * (gbest - position)
    return w * velocity + cognitive + social


def update_position(position, velocity, bounds):
    new_position = position + velocity
    return np.clip(new_position, bounds[0], bounds[1])


def update_pbest(positions, pbest_positions, pbest_scores):
    for i in range(len(positions)):
        score = objective_function(positions[i])
        if score < pbest_scores[i]:
            pbest_positions[i] = positions[i]
            pbest_scores[i] = score
    return pbest_positions, pbest_scores


def get_global_best(pbest_positions, pbest_scores):
    best_idx = np.argmin(pbest_scores)
    return pbest_positions[best_idx], pbest_scores[best_idx]


def pso(num_particles=30, max_iter=100, w=0.5, c1=2.0, c2=2.0, bounds=(-4.5, 4.5)):
    positions, velocities = initialize_particles(num_particles, bounds)
    pbest_positions = np.copy(positions)
    pbest_scores = np.array([objective_function(pos) for pos in positions])
    gbest_position, gbest_score = get_global_best(pbest_positions, pbest_scores)

    convergence = []

    for _ in range(max_iter):
        velocities = update_velocity(velocities, positions, pbest_positions, gbest_position, w, c1, c2)
        positions = update_position(positions, velocities, bounds)
        pbest_positions, pbest_scores = update_pbest(positions, pbest_positions, pbest_scores)
        gbest_position, gbest_score = get_global_best(pbest_positions, pbest_scores)
        convergence.append(gbest_score)

    return gbest_position, gbest_score, convergence


if __name__ == '__main__':
    test_configs = [
        {"num_particles": 2, "max_iter": 5, "w": 0.9, "c1": 1.5, "c2": 1.5},
        {"num_particles": 3, "max_iter": 300, "w": 0.8, "c1": 2.0, "c2": 2.5},
        {"num_particles": 5, "max_iter": 50, "w": 0.2, "c1": 0.5, "c2": 3.5},
        {"num_particles": 6, "max_iter": 20, "w": 0.1, "c1": 3.0, "c2": 0.1},
        {"num_particles": 7, "max_iter": 80, "w": 0.6, "c1": 1.0, "c2": 3.0},
        {"num_particles": 10, "max_iter": 100, "w": 0.95, "c1": 2.0, "c2": 2.0},
        {"num_particles": 12, "max_iter": 75, "w": 0.3, "c1": 2.5, "c2": 1.5},
        {"num_particles": 15, "max_iter": 200, "w": 0.5, "c1": 3.0, "c2": 0.5},
        {"num_particles": 20, "max_iter": 150, "w": 0.4, "c1": 1.5, "c2": 2.5},
        {"num_particles": 25, "max_iter": 1, "w": 0.5, "c1": 2.0, "c2": 2.0},
        {"num_particles": 30, "max_iter": 100, "w": 0.7, "c1": 1.5, "c2": 1.5},
        {"num_particles": 35, "max_iter": 60, "w": 0.85, "c1": 2.2, "c2": 1.2},
        {"num_particles": 40, "max_iter": 120, "w": 0.6, "c1": 1.5, "c2": 2.0},
        {"num_particles": 45, "max_iter": 90, "w": 0.2, "c1": 3.5, "c2": 1.0},
        {"num_particles": 50, "max_iter": 300, "w": 0.6, "c1": 1.5, "c2": 1.5},
        {"num_particles": 60, "max_iter": 200, "w": 0.5, "c1": 1.0, "c2": 2.0},
        {"num_particles": 70, "max_iter": 50, "w": 0.4, "c1": 1.2, "c2": 2.8},
        {"num_particles": 80, "max_iter": 250, "w": 0.3, "c1": 2.7, "c2": 0.7},
        {"num_particles": 90, "max_iter": 30, "w": 0.9, "c1": 0.1, "c2": 3.0},
        {"num_particles": 100, "max_iter": 500, "w": 0.7, "c1": 1.0, "c2": 1.0}
    ]

    results = []

    for x, cfg in enumerate(test_configs):
        g_best, scored, conv = pso(**cfg)
        results.append((cfg, g_best, scored))
        print(f"{x+1} — parametry: {cfg} -> Najlepsza pozycja: {g_best}, wynik: {scored:.4f}")

    best_result = min(results, key=lambda x: x[2])
    print("\nNajlepsza konfiguracja:")
    print(f"   Parametry: {best_result[0]}")
    print(f"   Pozycja: {best_result[1]}")
    print(f"   Wynik: {best_result[2]:.4f}")




