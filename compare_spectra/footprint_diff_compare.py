import h5py
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from collections import Counter
from contextlib import closing

def get_adjacent_footprints(i):
    if i < 0 or i >=8:
        return [-1]
    return np.unique([max(0,i-2), max(0,i-1), min(i+1,8), min(i+2,8)])

with closing(h5py.File('../oco2_L1bScND_13682a_170126_B7302_170127164246.h5', 'r')) as l1:
    data = np.array(l1['SoundingMeasurements']['radiance_weak_co2'][0])
    '../oco2_L1bScND_13682a_170126_B7302_170127164246.h5'

    footprint = 4

    adjacent_footprints = get_adjacent_footprints(footprint)

    diffs = {i:0 for i in range(len(data[footprint]))}

    for chan in range(len(data[footprint])):
        for fp in adjacent_footprints:
            diffs[chan] += (data[fp][chan] - data[footprint][chan])

    diff_mean = np.mean(diffs.values())
    diff_std  = np.std(diffs.values())

    bad_channels = []
    for i, c in enumerate(data[footprint]):
        print (c - diff_mean) / diff_std
        if (c - diff_mean) / diff_std > 3.0:
            bad_channels.append(i)

    print bad_channels
