import random
import heapq


def generate_cities(num_cities=10):
    cities = []
    for _ in range(num_cities):
        x = random.randint(-100, 100)
        y = random.randint(-100, 100)
        z = random.randint(0, 50)
        cities.append((x, y, z))

    print('-----cities-----')
    for city in cities:
        print(city)
    print('\n')

    return cities


def euclides_distance_3(city_a, city_b):
    distance = (((city_a[0] - city_b[0]) ** 2) + ((city_a[1] - city_b[1]) ** 2) + ((city_a[2] - city_b[2]) ** 2)) ** 0.5
    if city_a[2] > city_b[2]:
        distance *= 0.9
    if city_a[2] < city_b[2]:
        distance *= 1.1
    return distance


def randomize_matrix(matrix, rate=0.2):
    size = len(matrix)
    to_delete = round(((size**2-size)/2)*rate)
    for _ in range(to_delete):
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)
        while matrix[x][y] == 0 or matrix[x][y] == -1:
            x = random.randint(0, size-1)
            y = random.randint(0, size-1)
        matrix[x][y] = -1


def distance_matrix(cities):
    city_list = list(cities)
    size = len(city_list)
    matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = euclides_distance_3(city_list[i], city_list[j])

    randomize_matrix(matrix, rate=0.2)

    print('-----matrix-----')
    for row in matrix:
        print(row)

    return matrix


def my_dfs(matrix, start=0):
    num_cities = len(matrix)
    paths = [([start], 0)]
    best_path = None
    valid_path_found = False
    min_cost = float('inf')

    while paths:
        path, cost = paths.pop()
        city = path[-1]

        if len(path) == num_cities:
            cost += matrix[city][start]
            if matrix[city][start] != -1 and cost < min_cost:
                min_cost = cost
                best_path = path + [start]
                valid_path_found = True
            continue

        for next_city in range(num_cities):
            if next_city not in path and matrix[city][next_city] != -1:
                paths.append((path + [next_city], cost + matrix[city][next_city]))

    if valid_path_found:
        return best_path, min_cost
    return None, None


def my_bfs(matrix, start=0):
    num_cities = len(matrix)
    paths = [([start], 0)]
    best_path = None
    valid_path_found = False
    min_cost = float('inf')

    while paths:
        path, cost = paths.pop(0)
        city = path[-1]

        if len(path) == num_cities:
            cost += matrix[city][start]
            if matrix[city][start] != -1 and cost < min_cost:
                min_cost = cost
                best_path = path + [start]
                valid_path_found = True
            continue

        for next_city in range(num_cities):
            if next_city not in path and matrix[city][next_city] != -1:
                paths.append((path + [next_city], cost + matrix[city][next_city]))

    if valid_path_found:
        return best_path, min_cost
    return None, None


def greedy_nearest_neighbor(matrix, start=0):
    n = len(matrix)
    visited = [False] * n
    path = [start]
    total_cost = 0
    visited[start] = True
    current = start

    for _ in range(n - 1):
        next_city = None
        min_dist = float('inf')
        for j in range(n):
            if not visited[j] and matrix[current][j] != -1 and matrix[current][j] < min_dist:
                min_dist = matrix[current][j]
                next_city = j
        if next_city is None:
            return None, float('inf')
        visited[next_city] = True
        path.append(next_city)
        total_cost += min_dist
        current = next_city

    if matrix[current][start] == -1:
        return None, float('inf')
    path.append(start)
    total_cost += matrix[current][start]
    return path, total_cost


def greedy_farthest_insertion(matrix, start=0):
    n = len(matrix)
    path = [start]
    unvisited = list(range(n))
    unvisited.remove(start)

    max_dist = -1
    farthest = None
    for city in unvisited:
        if matrix[start][city] != -1 and matrix[start][city] > max_dist:
            max_dist = matrix[start][city]
            farthest = city
    path.append(farthest)
    unvisited.remove(farthest)

    while unvisited:
        max_dist = -1
        next_city = None
        for city in unvisited:
            min_dist = float('inf')
            for p in path:
                if matrix[city][p] != -1 and matrix[city][p] < min_dist:
                    min_dist = matrix[city][p]
            if min_dist > max_dist:
                max_dist = min_dist
                next_city = city

        best_pos = 0
        best_increase = float('inf')
        for i in range(len(path)):
            a = path[i]
            b = path[(i + 1) % len(path)]

            if matrix[a][next_city] != -1 and matrix[next_city][b] != -1:
                increase = matrix[a][next_city] + matrix[next_city][b] - matrix[a][b]

                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1

        path.insert(best_pos, next_city)
        path.insert(best_pos, next_city)
        unvisited.remove(next_city)

    path.append(start)
    cost = 0
    for i in range(len(path) - 1):
        cost += matrix[path[i]][path[i + 1]]

    return path, cost


def a_star_heuristic_admissible(matrix, start=0):
    size = len(matrix)
    queue = []

    def heuristic(current_city, unvisited):
        if not len(unvisited):
            return float('-inf')
        min_cost = float('inf')
        for city in unvisited:
            cost = matrix[current_city][city]
            if cost != -1 and cost < min_cost:
                min_cost = cost

        for i in unvisited:
            for j in unvisited:
                cost = matrix[i][j]
                if i != j and cost != -1 and cost < min_cost:
                    min_cost = cost
            if matrix[i][0] != -1 and i != 0:
                min_cost = min(min_cost, matrix[i][0])

        min_cost = min_cost * len(unvisited)

        if min_cost and len(unvisited):
            return min_cost
        else:
            return float('-inf')

    heapq.heappush(queue, (
        heuristic(start, set(range(size)) - {start}),
        0,
        start,
        [start]
    ))

    best_path = None
    lowest_cost = float('inf')

    while queue:
        est_total_cost, curr_cost, curr_city, path = heapq.heappop(queue)

        if est_total_cost >= lowest_cost:
            continue

        if len(path) == size:
            if matrix[curr_city][start] != -1:
                total_cost = curr_cost + matrix[curr_city][start]
                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    best_path = path + [start]
            continue

        for next_city in range(size):
            if next_city not in path and matrix[curr_city][next_city] != -1:
                new_cost = curr_cost + matrix[curr_city][next_city]
                unvisited = set(range(size)) - set(path) - {next_city}
                h = heuristic(next_city, unvisited)
                est_cost = new_cost + h
                if est_cost < lowest_cost:
                    heapq.heappush(queue, (est_cost, new_cost, next_city, path + [next_city]))

    return best_path, lowest_cost


def a_star_heuristic_inadmissible(matrix, start=0):
    size = len(matrix)
    queue = []

    def heuristic(current_city, unvisited):
        total = 0
        count = 0
        for city in unvisited.union({start}):
            cost = matrix[current_city][city]
            if cost != -1:
                total += cost
                count += 1
        return (total / count) * (len(unvisited) + 1) if count > 0 else float('inf')

    heapq.heappush(queue, (
        heuristic(start, set(range(size)) - {start}),
        0,
        start,
        [start]
    ))

    best_path = None
    lowest_cost = float('inf')

    while queue:
        est_total_cost, curr_cost, curr_city, path = heapq.heappop(queue)

        if est_total_cost >= lowest_cost:
            continue

        if len(path) == size:
            if matrix[curr_city][start] != -1:
                total_cost = curr_cost + matrix[curr_city][start]
                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    best_path = path + [start]
            continue

        for next_city in range(size):
            if next_city not in path and matrix[curr_city][next_city] != -1:
                new_cost = curr_cost + matrix[curr_city][next_city]
                unvisited = set(range(size)) - set(path) - {next_city}
                h = heuristic(next_city, unvisited)
                est_cost = new_cost + h
                if est_cost < lowest_cost:
                    heapq.heappush(queue, (est_cost, new_cost, next_city, path + [next_city]))

    return best_path, lowest_cost


def aco_tsp(matrix, num_ants=10, num_iterations=10, alpha=1, beta=5, rho=0.5, Q=100):
    import random

    num_cities = len(matrix)
    pheromone = [[1.0 for _ in range(num_cities)] for _ in range(num_cities)]
    best_path = None
    best_cost = float('inf')

    def choose_next_city(current_city, visited):
        probabilities = []
        total = 0.0
        for next_city in range(num_cities):
            if not visited[next_city] and matrix[current_city][next_city] != -1:
                tau = pheromone[current_city][next_city] ** alpha
                eta = (1.0 / matrix[current_city][next_city]) ** beta
                prob = tau * eta
                probabilities.append((next_city, prob))
                total += prob

        if total == 0:
            return None

        rand_val = random.uniform(0, total)
        cumulative = 0.0
        for city, prob in probabilities:
            cumulative += prob
            if cumulative >= rand_val:
                return city

        return None

    def construct_route_for_ant():
        start_city = random.randint(0, num_cities - 1)
        visited = [False] * num_cities
        ant_path = [start_city]
        visited[start_city] = True
        current = start_city
        ant_path_cost = 0.0

        for _ in range(num_cities - 1):
            next_city = choose_next_city(current, visited)
            if next_city is None:
                return None, float('inf')
            ant_path.append(next_city)
            ant_path_cost += matrix[current][next_city]
            visited[next_city] = True
            current = next_city

        if matrix[current][start_city] == -1:
            return None, float('inf')

        ant_path.append(start_city)
        ant_path_cost += matrix[current][start_city]
        return ant_path, ant_path_cost

    def update_pheromones(best_iteration_path, best_iteration_cost):
        for i in range(num_cities):
            for j in range(num_cities):
                pheromone[i][j] *= (1.0 - rho)

        deposit = Q / best_iteration_cost
        for i in range(len(best_iteration_path) - 1):
            a, b = best_iteration_path[i], best_iteration_path[i + 1]
            pheromone[a][b] += deposit
            pheromone[b][a] += deposit

    for _ in range(num_iterations):
        iteration_best_path = None
        iteration_best_cost = float('inf')

        for _ in range(num_ants):
            path, cost = construct_route_for_ant()
            if path and cost < iteration_best_cost:
                iteration_best_path = path
                iteration_best_cost = cost
            if path and cost < best_cost:
                best_path = path
                best_cost = cost

        if iteration_best_path:
            update_pheromones(iteration_best_path, iteration_best_cost)

    return best_path, best_cost
