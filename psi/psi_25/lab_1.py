from functions import *

if __name__ == '__main__':
    num_of_cities = 10
    cities = generate_cities(num_of_cities)
    matrix = distance_matrix(cities)

    print('\nResults:')
    bfs_path, bfs_cost = my_bfs(matrix)
    if bfs_path:
        print("BFS Best Path:", bfs_path, "with Cost:", bfs_cost)
    else:
        print('BFS Best Path not found')

    dfs_path, dfs_cost = my_dfs(matrix)
    if dfs_path:
        print("DFS Best Path:", dfs_path, "with Cost:", dfs_cost)
    else:
        print('DFS Best Path not found')