import h5py
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import sys

from collections import Counter
from contextlib import closing

if len(sys.argv) != 4:
    sys.exit("ARGUEMENT ERROR\n Usage: python plot_footprints <band> <sample> <footprint>")

# band       - [0,1,2]     - correspond to o2, weak_co2, strong_co2
# sample    - [0 - ~8100] - index of sample in file
# footprint - [0 - 8]     - index of footprint in sample

bands = {0:'radiance_o2',
         1:'radiance_weak_co2',
         2:'radiance_strong_co2'}

band      = int(sys.argv[1])
sample    = int(sys.argv[2])
footprint = int(sys.argv[3])

with closing(h5py.File('../oco2_L1bScND_13682a_170126_B7302_170127164246.h5', 'r')) as l1:
    if footprint == -1:
        data = np.array(l1['SoundingMeasurements'][bands[band]][sample])

        f, axarr = plt.subplots(2,4)
        for i in range(8):
            axarr[i/4][i%4].plot(data[i])
            axarr[i/4][i%4].set_title('Footprint ' + str(i))
            axarr[i/4][i%4].xaxis.set_ticks([])
            axarr[i/4][i%4].yaxis.set_ticks([])
    else:
        data = np.array(l1['SoundingMeasurements'][bands[band]][sample][footprint])
        plt.plot(data)

    plt.show()
