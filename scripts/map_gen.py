import numpy as np
import matplotlib.pyplot as plt


# def gen_spice_map(width: int, height: int, n_heaps: int):
#     # Initialize an empty map
#     spice_map = np.zeros((width, height))
#
#     # Generate random positions for the heaps
#     heap_pos_x = np.random.randint(0, width, n_heaps)
#     heap_pos_y = np.random.randint(0, height, n_heaps)
#
#     # Define the binomial distribution parameters
#     n_trials = 20  # Number of trials (can be adjusted)
#     p_success = 0.6  # Probability of success (can be adjusted)
#
#     # Generate and place the heaps
#     for (x, y) in zip(heap_pos_x, heap_pos_y):
#         # Generate a binomial distribution
#         heap_size = np.random.binomial(n_trials, p_success)
#
#         # Place the heap centered around (x, y)
#         for i in range(-heap_size, heap_size + 1):
#             for j in range(-heap_size, heap_size + 1):
#                 if 0 <= x + i < width and 0 <= y + j < height:
#                     spice_map[x + i, y + j] += np.random.binomial(n_trials, p_success)
#
#     return spice_map


def gen_spice_map(width: int, height: int, n_heaps: int, total_spice: int):
    # Initialize an empty map
    spice_map = np.zeros((width, height))
    heap_pos_x = np.random.randint(0, width, n_heaps)
    heap_pos_y = np.random.randint(0, height, n_heaps)

    for (heap_x, heap_y) in zip(heap_pos_x, heap_pos_y):
        # Generate a random covariance matrix to add variability
        cov = np.random.uniform(30, 90, (2, 2))
        cov = cov @ cov.T
        heap = np.random.multivariate_normal([heap_x, heap_y], cov, size=total_spice).astype(int)
        for (x, y) in zip(heap[:, 0], heap[:, 1]):
            if 0 < x < width and 0 < y < height:
                spice_map[x, y] += 1
    return (spice_map / np.max(spice_map) * 20)


# Example usage
width = 1000
height = 1000
n_heaps = 5
total_spice = 10000
spice_map = gen_spice_map(width, height, n_heaps, total_spice)


plt.imshow(spice_map, cmap='terrain', interpolation='nearest')
plt.colorbar()
plt.show()
