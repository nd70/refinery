import numpy as np


# read through the full output and combine into an array
with open('data.txt') as f:
    lines = f.readlines()

out = np.zeros((len(lines), 4), dtype=object)
for ix, line in enumerate(lines):
    temp = line.strip('\n').split(' ')
    if len(temp) == 4:
        out[ix, :] = temp

c2 = np.array([np.float(x) for x in out[:, 2]])
c3 = np.array([np.float(x) for x in out[:, 3]])
out[:, 2] = c2
out[:, 3] = c3

# get rid of empty entries
drop = []
for ii in range(out.shape[0]):
    if out[ii, 2] == 0 and out[ii, 3] == 0:
        drop.append(ii)
out = np.delete(out, drop, 0)

# sort by: number of bins with a clean/darm power ratio of 0.9 or more (col 3)
# - OR - sort by the bin with the most subtraction (col 2)
sorted_by_sub = out[out[:,2].argsort()]
sorted_by_bins = out[out[:,3].argsort()]

top_50_sub = sorted_by_sub[:50, :]
top_50_bins = sorted_by_bins[-50:, :]

with open('top_50_sub.txt', 'w') as f:
    for ii in range(50):
        f.write('{}\n'.format(top_50_sub[ii, :]))

with open('top_50_bins.txt', 'w') as f:
    for ii in range(50):
        f.write('{}\n'.format(top_50_bins[ii, :]))
