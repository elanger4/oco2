import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

from collections import OrderedDict
from contextlib import closing

def get_bad_channels_polyfit(sample, num_std, _degree):
    bad_channels = []

    coefs = np.polyfit(np.arange(len(sample)),
                          sample,
                          _degree)
    fitted_line = OrderedDict([])
    for i, dat in enumerate(sample):
        x = [i ** d for d in range(_degree+1)]

        fitted_line[i] = np.sum(coefs * x[::-1])

    new_std = np.std(fitted_line.values())
    
    for (i, dat), fit in zip(enumerate(sample), fitted_line.values()):
        if abs(dat - fit) / new_std > num_std:
            bad_channels.append(i)

    #print bad_channels
    return bad_channels

bad_samples = []
with open('bad_sample_locations.txt') as f:
    bad_samples = f.read().splitlines()

    bad_samples = [(int(s[0]), int(s[2])) for s in bad_samples]


bad_lookup ={0:'radiance_o2', 1:'radiance_weak_co2', 2:'radiance_strong_co2'}


for sam in bad_samples:
    with closing(h5py.File('oco2_L1bScND_13682a_170126_B7302_170127164246.h5', 'r')) as l1:
        
        samples = np.array(l1['SoundingMeasurements'][bad_lookup[sam[0]]])
        mask = np.array(l1['InstrumentHeader']['snr_coef'][:, :, :, 2])
        
        # This gets the proper footprint
        sample = samples[0][sam[1]] # First sample of all
        _mask    = mask[sam[0]][sam[1]]


        bad_channels = get_bad_channels_polyfit(sample, 4.5, 1)

        _degree = 25
        plt.plot(sample)
        coefs = np.polyfit(np.arange(len(sample)),
                              sample,
                              _degree)
        fitted_line = OrderedDict([])
        for i, dat in enumerate(sample):
            x = [i ** d for d in range(_degree+1)]

            fitted_line[i] = np.sum(coefs * x[::-1])

        plt.plot(fitted_line.values(), color='blue')

        p1 = False
        p2 = False
        p4 = False
        bc = False
        cb = False


        for (i, val), msk in zip(enumerate(sample), _mask):
            if msk == 1.0:
                if not p1:
                    plt.scatter(i, val, color='orange', label='Radiometric Problem')
                    p1 = True
                else:
                    plt.scatter(i, val, color='orange')
                    
            elif msk == 2.0:
                if not p2:
                    plt.scatter(i, val, color='blue', label='Spatial Problem')
                    p2 = True
                else:
                    plt.scatter(i, val, color='blue')

            elif msk == 4.0:
                if not p4:
                    plt.scatter(i, val, color='green', label='Spectral Problem')
                    p4 = True
                else:
                    plt.scatter(i, val, color='green')
            if i in bad_channels and msk != 1.0:
                if not bc:
                    plt.scatter(i, val, color='yellow', label='Incorrectly Labeled as bad channel')
                    bc = True
                else:
                    plt.scatter(i, val, color='yellow')
            if i in bad_channels and msk == 1.0:
                print 'WE DID IT!!!!'
                if not cb:
                    plt.scatter(i, val, color='red', label='Correctly caught bad sample')
                    cb = True
                else:
                    plt.scatter(i, val, color='red')
            


        plt.legend()
        plt_name =  'polyfit_reak_peak_finder_' + str(sam[0]) + '_' + str(sam[1]) + '.png'
        print plt_name
        plt.savefig(plt_name)
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()
