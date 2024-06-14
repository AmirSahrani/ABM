import numpy as np
from model import DuneModel


def gen_spice_map(width: int, height: int, n_heaps: int, total_spice: int, cov_range: tuple= (3,9)):
    spice_map = np.zeros((width, height))
    heap_pos_x = np.random.randint(0, width, n_heaps)
    heap_pos_y = np.random.randint(0, height, n_heaps)

    for (heap_x, heap_y) in zip(heap_pos_x, heap_pos_y):
        cov = np.random.uniform(cov_range[0], cov_range[1], (2, 2))
        cov = cov @ cov.T
        heap = np.random.multivariate_normal([heap_x, heap_y], cov, size=total_spice).astype(int)

        for (x, y) in zip(heap[:, 0], heap[:, 1]):
            if 0 < x < width and 0 < y < height:
                spice_map[x, y] += 1

    return (spice_map / np.max(spice_map) * 20).astype(int)


def gen_river(width, height):
    river = np.zeros((width, height))
    river[width // 2, :] = 1
    return river


def random_locations(width: int, height: int, model: DuneModel):
    return zip(np.random.randint(0, width, model.n_agents), np.random.randint(0, height, model.n_agents))
