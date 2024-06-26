import numpy as np
from model import DuneModel
import pandas as pd


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
                
    normalization_factor = total_spice * n_heaps / np.sum(spice_map)
    final = (spice_map * normalization_factor).astype(int)
    return final


def gen_river_line(model: DuneModel):
    width, height = model.width, model.height
    river = np.zeros((width, height))
    river[width // 2, :] = 1
    return river


def no_river(model: DuneModel):
    width, height = model.width, model.height
    return np.zeros((width, height))


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
        river[loc[0] % width, loc[1]] = 1
        loc += directions[np.random.randint(3)]
    return river


def random_locations(model: DuneModel):
    width, height = model.width, model.height
    return zip(np.random.randint(0, width, model.n_agents), np.random.randint(0, height, model.n_agents))


def split_tribes_locations(model: DuneModel):
    width, height = model.width, model.height

    left_bound = 0 + (width // model.n_tribes) * (len(model.tribes) - 1)
    right_bound = left_bound + width // model.n_tribes

    top_bound = 0 + (height // model.n_tribes) * (len(model.tribes) - 1)
    bottom_bound = top_bound + height // model.n_tribes

    return zip(np.random.randint(left_bound,
                                 right_bound, model.n_agents),
               np.random.randint(top_bound,
                                 bottom_bound, model.n_agents))


def tribe_locations_naturally_distributed(model: DuneModel):
    width, height = model.width, model.height
    n_tribes = model.n_tribes
    agents_per_tribe = model.n_agents // n_tribes
    cov_range = model.spice_kwargs["cov_range"]
    
    tribe_centers = np.column_stack((
        np.random.randint(0, width, n_tribes),
        np.random.randint(0, height, n_tribes)
    ))

    locations = []
    for center_x, center_y in tribe_centers:
        cov_value = np.random.uniform(cov_range[0], cov_range[1])
        cov = np.array([[cov_value, 0], [0, cov_value]])  
        tribe = np.random.multivariate_normal([center_x, center_y], cov, size=agents_per_tribe).astype(int)
        tribe = np.clip(tribe, [0, 0], [width - 1, height - 1])
        locations.extend(tribe)

    locations = np.array(locations)  

    return zip(locations[:, 0], locations[:, 1])



# def tribe_locations_single_cluster_per_tribe(model: DuneModel):
#     width, height = model.width, model.height
#     n_tribes = model.n_tribes//model.n_tribes
#     agents_per_tribe = model.n_agents // n_tribes
#     cov_range = model.spice_kwargs["cov_range"]

#     tribe_centers = np.column_stack((
#         np.random.randint(0, width, n_tribes),
#         np.random.randint(0, height, n_tribes)
#     ))

#     locations = []
#     for center_x, center_y in tribe_centers:
#         cov_value = np.random.uniform(cov_range[0], cov_range[1])
#         cov = np.array([[cov_value, 0], [0, cov_value]])
#         tribe = np.random.multivariate_normal([center_x, center_y], cov, size=agents_per_tribe).astype(int)
#         tribe = np.clip(tribe, [0, 0], [width - 1, height - 1])
#         locations.extend(tribe)

#     locations = np.array(locations)

#     return zip(locations[:, 0], locations[:, 1])


def tribe_locations_single_cluster_per_tribe(model: DuneModel):
    width, height = model.width, model.height
    n_tribes = model.n_tribes//model.n_tribes
    agents_per_tribe = model.n_agents // n_tribes
    cov_range = model.spice_kwargs["cov_range"]

    tribe_centers = np.column_stack((
        np.random.randint(0, width, n_tribes),
        np.random.randint(0, height, n_tribes)
    ))

    occupied_positions = set()
    locations = []
    for center_x, center_y in tribe_centers:
        cov_value = np.random.uniform(cov_range[0], cov_range[1])
        cov = np.array([[cov_value, 0], [0, cov_value]])
        tribe = np.random.multivariate_normal([center_x, center_y], cov, size=agents_per_tribe).astype(int)
        tribe = np.clip(tribe, [0, 0], [width - 1, height - 1])
        
        for pos in tribe:
            position = tuple(pos)
            while position in occupied_positions:
                position = (np.random.randint(0, width), np.random.randint(0, height))
            occupied_positions.add(position)
            locations.append(position)

    return locations

def gen_central_spice_heap(model: DuneModel):
    width, height = model.width, model.height
    total_spice = model.spice_kwargs["total_spice"]
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4 
    spice_map = np.zeros((width, height))

    angles = np.random.uniform(0, 2 * np.pi, total_spice)
    radii = np.random.uniform(0, radius, total_spice)

    x_offsets = radii * np.cos(angles)
    y_offsets = radii * np.sin(angles)

    spice_positions_x = (center_x + x_offsets).astype(int)
    spice_positions_y = (center_y + y_offsets).astype(int)

    for (x, y) in zip(spice_positions_x, spice_positions_y):
        if 0 <= x < width and 0 <= y < height:
            spice_map[x, y] += 1

    normalization_factor = total_spice / np.sum(spice_map)
    final = (spice_map * normalization_factor).astype(int)
    return final

