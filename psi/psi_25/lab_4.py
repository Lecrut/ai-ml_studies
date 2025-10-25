from functions import *
import time

if __name__ == '__main__':
    num_of_cities = 8
    cities = generate_cities(num_of_cities)
    matrix = distance_matrix(cities)

    print()

    start = time.perf_counter()
    best_path_aco, best_cost_aco = aco_tsp(matrix, num_ants=20, num_iterations=10)
    end = time.perf_counter()
    aco_time = (end - start) * 1000

    print("\nACO Solution")
    if best_path_aco:
        print(f" Path: {best_path_aco}")
        print(f" Cost: {best_cost_aco:.2f}")
        print(f" Time: {aco_time:.2f} ms\n")
    else:
        print("No valid path found.")

    city_start = best_path_aco[0] if best_path_aco else 0

    start = time.perf_counter()
    dfs_path, dfs_cost = my_dfs(matrix, city_start)
    end = time.perf_counter()
    dfs_time = (end - start) * 1000

    if dfs_path:
        print("DFS (full search)")
        print(f"  Path: {dfs_path}")
        print(f"  Cost: {dfs_cost:.2f}")
        print(f"  Time: {dfs_time:.2f} ms\n")
    else:
        print('DFS Best Path not found')

    start = time.perf_counter()
    bfs_path, bfs_cost = my_bfs(matrix, city_start)
    end = time.perf_counter()
    bfs_time = (end - start) * 1000

    if bfs_path:
        print("BFS")
        print(f' Path: {bfs_path}')
        print(f" Cost: {bfs_cost: .2f}")
        print(f"  Time: {dfs_time:.2f} ms\n")
    else:
        print('BFS Best Path not found')

    start = time.perf_counter()
    nn_path, nn_cost = greedy_nearest_neighbor(matrix, city_start)
    end = time.perf_counter()
    nn_time = (end - start) * 1000

    print("Nearest Neighbour")
    print(f"  Path: {nn_path}")
    print(f"  Cost: {nn_cost:.2f}")
    print(f"  Time: {nn_time:.2f} ms\n")

    start = time.perf_counter()
    a_star_path, a_star_cost = a_star_heuristic_admissible(matrix, city_start)
    end = time.perf_counter()
    a_star_time = (end - start) * 1000

    print("A Start with admissible")
    print(f"  Path: {a_star_path}")
    print(f"  Cost: {a_star_cost:.2f}")
    print(f"  Time: {a_star_time:.2f} ms\n")

    start = time.perf_counter()
    a_star_2_path, a_star_2_cost = a_star_heuristic_inadmissible(matrix, city_start)
    end = time.perf_counter()
    a_star_2_time = (end - start) * 1000

    print("A Start with inadmissible")
    print(f"  Path: {a_star_2_path}")
    print(f"  Cost: {a_star_2_cost:.2f}")
    print(f"  Time: {a_star_2_time:.2f} ms\n")

