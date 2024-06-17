import numpy as np
from model import DuneModel


def gen_spice_map(model: DuneModel):
    width, height, n_heaps = model.width, model.height, model.n_heaps
    total_spice = model.spice_kwargs["total_spice"]
    cov_range = model.spice_kwargs["cov_range"]
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


def gen_river_line(model: DuneModel):
    width, height = model.width, model.height
    river = np.zeros((width, height))
    river[width // 2, :] = 1
    return river


def no_river(model: DuneModel):
    width, height = model.width, model.height
    return  np.zeros((width, height))

def gen_river_random(model: DuneModel):
    """ Generates a river using a random walker """
    width, height = model.width, model.height
    river = np.zeros((width, height))
    loc = np.array([width // 2, 0])
    directions = [
        np.array([-1, 0]),
        np.array([1, 0]),
        np.array([0, 1])
    ]

    river[loc[0], loc[1]] = 1
    while loc[1] != height:
        river[loc[0], loc[1]] = 1
        loc += directions[np.random.randint(3)]
    return river


def random_locations(model: DuneModel):
    width, height = model.width, model.height
    return zip(np.random.randint(0, width, model.n_agents), np.random.randint(0, height, model.n_agents))


def split_tribes_locations(model: DuneModel):
    width, height = model.width, model.height
    """ Assumes only two tribes, locations will be top left, bottom right"""

    assert model.n_tribes == 2, "Splitting tribes only works for 2 tribes"
    left_bound = 0 + (width // 2) * (len(model.tribes) - 1)
    right_bound = left_bound + width // 2

    top_bound = 0 + (height // 2) * (len(model.tribes) - 1)
    bottom_bound = top_bound + height // 2 

    return zip(np.random.randint(left_bound,
                                 right_bound, model.n_agents),
               np.random.randint(top_bound,
                                 bottom_bound, model.n_agents))