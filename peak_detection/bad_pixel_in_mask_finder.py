import h5py
import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict, Counter
from contextlib import closing

bad_pixels = []
with closing(h5py.File('oco2_L1bScND_13682a_170126_B7302_170127164246.h5', 'r')) as l1:
    samples = np.array(l1['SoundingMeasurements']['radiance_o2'])
    mask = np.array(l1['InstrumentHeader']['snr_coef'][:, :, :, 2])
    print np.shape(mask)

    for i, band in enumerate(mask):
        for j, footp in enumerate(band):
            if Counter(footp)[1.0] > 0:
                bad_pixels.append(((i,j), Counter(footp)[1.0]) )

fp = open('bad_channel_locations.txt', 'w')

for p in bad_pixels:
    fp.write(str(p[0][0]) + ' ' + str(p[0][1]) + '\n')


                
