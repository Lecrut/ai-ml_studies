import numpy as np
import matplotlib.pyplot as plt

# Środek rozkładu (może być 0,0)
mean = [-4, -4]

# Macierze kowariancji
A = [[2, 0], [0, 2]]
B = [[3, -1], [-1, 1]]
C = [[2, 5], [5, 2]]

# Tworzymy dane
# points_A = np.random.multivariate_normal(mean, A, 1)
# points_B = np.random.multivariate_normal(mean, B, 50)
points_C = np.random.multivariate_normal(mean, C, 5)

# Rysujemy
plt.figure(figsize=(8, 8))
# plt.scatter(*points_A.T, alpha=0.5, label='A', color='blue')
# plt.scatter(*points_B.T, alpha=0.5, label='B', color='green')
plt.scatter(*points_C.T, alpha=0.5, label='C', color='red')
plt.legend()
plt.axis('equal')
plt.title('Zbiory punktów o różnych macierzach kowariancji')
plt.grid(True)
plt.show()
