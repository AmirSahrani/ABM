import numpy as np
import matplotlib.pyplot as plt
from glob import glob


files = [f for f in glob("*.npy")]
files.remove("final_amir_laptop_23.npy")
files.remove('intermediate_amir_laptop.npy')
arrays = [np.load(x) for x in files]

small = 'intermediate_amir_laptop.npy'
small_array = np.load(small)
mask = np.any(small_array != 0, axis=1)
small_array_filter = small_array[mask]
print(small_array_filter.shape)

# small_arrays[1][:23, ...] = 0

# print(small_arrays[1].shape)


last = [np.sum(small_arrays, axis=0)]
# print(np.max(last), np.max(small_arrays[0]), np.max(small_arrays[1]))


first = [x for x in arrays if x.shape[0] == 4608]
[last.append(x) for x in arrays if x.shape[0] != 4608]

second = np.vstack(last)

big_array = np.sum(first, axis=0)
np.save('final_512.npy', big_array)


# stacked = np.vstack([big_array, second])
# plt.yscale('log')
# plt.plot(stacked[:, 0, -1], 'o')
# plt.show()


