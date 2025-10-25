from functions import *
import time


if __name__ == '__main__':
    num_of_cities = 10
    cities = generate_cities(num_of_cities)
    matrix = distance_matrix(cities)

    # DFS (Full Search)
    start = time.perf_counter()
    dfs_path, dfs_cost = my_dfs(matrix)
    end = time.perf_counter()
    dfs_time = (end - start) * 1000  # ms

    print()

    print("DFS (full search)")
    print(f"  Path: {dfs_path}")
    print(f"  Cost: {dfs_cost:.2f}")
    print(f"  Time: {dfs_time:.2f} ms\n")

    # Nearest Neighbour
    start = time.perf_counter()
    nn_path, nn_cost = greedy_nearest_neighbor(matrix)
    end = time.perf_counter()
    nn_time = (end - start) * 1000

    print("Nearest Neighbour")
    print(f"  Path: {nn_path}")
    print(f"  Cost: {nn_cost:.2f}")
    print(f"  Time: {nn_time:.2f} ms\n")

    # Cheapest Insertion
    start = time.perf_counter()
    tfs_path, tfs_cost = greedy_farthest_insertion(matrix)
    end = time.perf_counter()
    bf_time = (end - start) * 1000

    print("TFS")
    print(f"  Path: {tfs_path}")
    print(f"  Cost: {tfs_cost:.2f}")
    print(f"  Time: {bf_time:.2f} ms\n")