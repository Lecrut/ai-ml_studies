import pandas as pd
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


def load_tasks_from_excel(filepath, sheet_name="Arkusz2"):
    xls = pd.read_excel(filepath, sheet_name=sheet_name)
    xls.head(10)
    df_cleaned = xls.iloc[1:].reset_index(drop=True)

    tasks = []
    next_col = 1
    for i in range(1, 51):
        task = []
        for row in range(0, 11):
            resource = (df_cleaned[i][row], df_cleaned[f"Unnamed: {next_col}"][row])
            task.append(resource)
        next_col += 2
        tasks.append(task)

    return tasks


def decode_schedule(tasks, chromosome):
    machine_available = defaultdict(int)
    task_end_time = defaultdict(int)
    schedule = []

    for task_id, op_idx in chromosome:
        resource, duration = tasks[task_id][op_idx]
        start = max(machine_available[resource], task_end_time[task_id])
        end = start + duration
        machine_available[resource] = end
        task_end_time[task_id] = end
        schedule.append((task_id, op_idx, resource, start, end))

    makespan = max(end for *_, end in schedule)
    return makespan, schedule


def create_population(tasks, size):
    operations = [(task_id, op_idx)
                  for task_id, task in enumerate(tasks)
                  for op_idx in range(len(task))]
    return [random.sample(operations, len(operations)) for _ in range(size)]


def fitness(tasks, individual):
    return decode_schedule(tasks, individual)[0]


def selection(tasks, population, k=1):
    return min(random.sample(population, k), key=lambda ind: fitness(tasks, ind))


def crossover(p1, p2, num_operations, num_tasks):
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start:end] = p1[start:end]

    idx = 0
    for gene in p2:
        if gene not in child:
            while child[idx] is not None:
                idx += 1
            child[idx] = gene

    counts = Counter(child)

    excess_indices = [i for i, task in enumerate(child) if counts[task] > num_operations]

    missing_tasks = []
    for task in range(num_tasks):
        missing = num_operations - counts[task]
        missing_tasks.extend([task] * missing)

    for i, new_task in zip(excess_indices, missing_tasks):
        old_task = child[i]
        child[i] = new_task
        counts[old_task] -= 1
        counts[new_task] += 1

    return child


def smart_mutate(individual, rate=0.15):
    for i in range(len(individual) - 1):
        if random.random() < rate:
            individual[i], individual[i + 1] = individual[i + 1], individual[i]


def genetic_algorithm(tasks, generations=150, pop_size=100, mutation_rate=0.15, elite_fraction=0.2):
    elite_size = max(1, int(elite_fraction * pop_size))
    population = create_population(tasks, pop_size)
    best_fit = []
    best_individual = None
    best_score = float('inf')

    num_tasks = len(tasks)
    num_operations = len(tasks[0])

    for gen in range(generations):
        population = sorted(population, key=lambda ind: fitness(tasks, ind))
        elites = population[:elite_size]

        new_population = elites[:]
        while len(new_population) < pop_size:
            parent1 = selection(tasks, population)
            parent2 = selection(tasks, population)
            child = crossover(parent1, parent2, num_operations, num_tasks)
            smart_mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population
        current_best = min(population, key=lambda ind: fitness(tasks, ind))
        current_score = fitness(tasks, current_best)
        best_fit.append(current_score)

        if current_score < best_score:
            best_score = current_score
            best_individual = current_best

        print(f"Generacja {gen + 1}: najlepszy makespan = {current_score}")

    return best_individual, best_score, best_fit


if __name__ == "__main__":
    filepath = "Datasets/GA_task.xlsx"
    tasks = load_tasks_from_excel(filepath)

    best_solution, best_makespan, history = genetic_algorithm(
        tasks,
        generations=50,
        pop_size=50,
        mutation_rate=0.3,
        elite_fraction=0.3
    )

    print(f"\nNajlepszy makespan: {best_makespan}")

    plt.plot(history)
    plt.title("Algorytm genetyczny")
    plt.xlabel("Generacja")
    plt.ylabel("Makespan")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # _, final_schedule = decode_schedule(tasks, best_solution)
    # print("\n📋 Przykładowe operacje z końcowego harmonogramu:")
    # for entry in final_schedule[:10]:
    #     print(f"Zadanie {entry[0]}, Operacja {entry[1]}: Maszyna {entry[2]}, Start {entry[3]}, Koniec {entry[4]}")
