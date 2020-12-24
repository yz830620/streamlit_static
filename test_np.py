import matplotlib.pyplot as plt
import numpy as np

samples = np.random.normal(0,100,10000)

hist, bins = np.histogram(samples, bins=20)

thresh = -200

mask = bins < thresh
below_thresh = np.array(bins[mask].tolist() + [thresh])

plt.figure(figsize=(10,6))

# original histogram
plt.bar(bins[:-1],hist, width=np.diff(bins), color='C0', align='edge');

# below threshold
plt.bar(below_thresh[:-1], hist[mask[:-1]], 
        width=np.diff(below_thresh), color='C1',
        align='edge')
plt.show()