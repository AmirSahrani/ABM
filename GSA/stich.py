import numpy as np
import matplotlib.pyplot as plt
from glob import glob


files = [f for f in glob("GSA/*.npy")]
print(files)
arrays = [np.load(x) for x in files if 'part_2' not in x and 'laptop' not in x]
# arrays.extend([np.load('GSA/final_amir_pc.npy')])
arrays_part_2 = [np.load(x) for x in files if 'part_2' in x and 'laptop' not in x]

all_arrays = np.sum(arrays, axis=0)
all_arrays_2 = np.sum(arrays_part_2, axis=0)
all = np.vstack((all_arrays, all_arrays_2))

print(all.shape)
mask = all!= 0

plt.plot(mask[mask.shape[0]//2:, :, -1])
plt.plot(mask[:mask.shape[0]//2, :, -1])
plt.show()

np.save('final_1024.npy', all)


plt.yscale('log')
plt.plot(all[:, 0, 0], 'o')
plt.show()
