import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


def create_fuzzy_system():
    distance = ctrl.Antecedent(np.arange(0, 210, 10), 'distance')
    angel = ctrl.Antecedent(np.arange(0, 91, 5), 'angel')
    resistance = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'resistance')
    mass = ctrl.Antecedent(np.arange(0.1, 10.1, 0.5), 'mass')
    v0 = ctrl.Consequent(np.arange(0, 100, 1), 'v0')

    distance.automf(3)
    angel.automf(3)
    resistance.automf(3)
    mass.automf(3)

    v0['low'] = fuzz.trimf(v0.universe, [0, 10, 40])
    v0['medium'] = fuzz.trimf(v0.universe, [30, 50, 70])
    v0['high'] = fuzz.trimf(v0.universe, [60, 90, 100])

    rules = [
        ctrl.Rule(distance['average'] & angel['average'] & resistance['poor'] & mass['average'], v0['low']),
        ctrl.Rule(distance['good'] & resistance['poor'] & mass['average'] & angel['average'], v0['medium']),
        ctrl.Rule(distance['good'] & resistance['good'], v0['high']),
        ctrl.Rule(distance['average'] & angel['average'] & resistance['average'], v0['medium']),
        ctrl.Rule(distance['poor'] & resistance['poor'], v0['low']),
        ctrl.Rule(angel['poor'] | angel['good'], v0['high']),
        ctrl.Rule(mass['good'] & (resistance['average'] | resistance['good']), v0['high']),
        ctrl.Rule(mass['average'] & resistance['good'], v0['medium']),
        ctrl.Rule(mass['poor'] & resistance['poor'], v0['low']),
        ctrl.Rule(distance['good'] & angel['average'] & mass['poor'] & resistance['average'], v0['medium']),
    ]

    system = ctrl.ControlSystem(rules)
    simulation = ctrl.ControlSystemSimulation(system)
    return simulation


def theoretical_velocity(d, a):
    g = 9.81
    alpha_rad = np.radians(a)
    return np.sqrt((d * g) / np.sin(2 * alpha_rad))


if __name__ == "__main__":
    test_cases = [
        (100, 45, 0.1, 1),
        (150, 45, 0.1, 1),
        (80, 30, 0.2, 2),
        (60, 60, 0.05, 1),
        (120, 50, 0.3, 3),
        (50, 45, 0.1, 1),
        (100, 30, 0.2, 1),
        (100, 60, 0.05, 1),
        (200, 45, 0.0, 1),
        (100, 45, 0.5, 5),
    ]

    sim = create_fuzzy_system()
    results = []

    for distance_val, angel_val, resistance_val, mass_val in test_cases:
        sim.input['distance'] = distance_val
        sim.input['angel'] = angel_val
        sim.input['resistance'] = resistance_val
        sim.input['mass'] = mass_val
        sim.compute()

        v_fuzzy = sim.output['v0']
        v_real = theoretical_velocity(distance_val, angel_val)
        error = abs(v_fuzzy - v_real) / v_real * 100

        results.append((distance_val, angel_val, resistance_val, mass_val, v_real, v_fuzzy, error))

    print(f"{'distance':>5} {'angel':>5} {'resistance':>5} {'mass':>5} | {'v0_ideal':>10} {'v0_fuzzy':>10} {'error [%]':>10}")
    print("-"*60)
    for r in results:
        print(f"{r[0]:5} {r[1]:5} {r[2]:5.2f} {r[3]:5.1f} | {r[4]:10.2f} {r[5]:10.2f} {r[6]:10.2f}")

    mean_error = np.mean([r[6] for r in results])
    print(f"\nŚredni błąd modelu rozmytego: {mean_error:.2f}%")

    v_real_values = [r[4] for r in results]
    v_fuzzy_values = [r[5] for r in results]
    labels = [f"{r[0]}/{r[1]}" for r in results]

    x = np.arange(len(results))

    plt.figure(figsize=(12, 6))
    plt.plot(x, v_real_values, marker='o', label='Teoretyczna v0')
    plt.plot(x, v_fuzzy_values, marker='x', label='Rozmyta v0')
    plt.xticks(x, labels, rotation=45)
    plt.xlabel('Przypadek testowy (distance/angel)')
    plt.ylabel('Prędkość początkowa [m/s]')
    plt.title('Porównanie: prędkość teoretyczna vs. rozmyta')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
