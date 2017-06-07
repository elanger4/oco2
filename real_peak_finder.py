import numpy as np
'''
import pandas as pd

data = pd.read_hdf('ocoL1_data.h5')

print data
'''
import h5py
import matplotlib.pyplot as plt

from collections import OrderedDict
from contextlib import closing

def get_bad_channels_zscore(sample, k, num_std):

    bad_channels = []
    
    for i, dat in enumerate(sample):

        # Select k values before and after specified point
        # and calculate Z-score
        sub_sample = sample[max(0,i-k):min(i+k+1,len(sample)+1)]
        sub_sample = np.delete(sub_sample, k)
            
        if np.abs((dat - np.mean(sub_sample))) / np.std(sub_sample) > num_std:
            bad_channels.append(i)

    return bad_channels

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

with closing(h5py.File('oco2_L1bScND_13682a_170126_B7302_170127164246.h5', 'r')) as l1:
    samples = np.array(l1['SoundingMeasurements']['radiance_o2'])
    mask = np.array(l1['InstrumentHeader']['snr_coef'][:, :, :, 2])
    footprint = 7
    fourth_band = samples[0][footprint]
    fourth_mask = mask[0][footprint]

    num_channels = len(fourth_band)

    p1 = False
    p2 = False
    p4 = False
    bc = False
    cb = False

    
    '''
    deg_std = (0,0)
    accuracy = 0
    for std in np.arange(2,5, 0.25):
        for d in range(1, 10):
            bad_pred = get_bad_channels_polyfit(fourth_band, std, d)

            if bad_pred:
                deg_std = (d, std)
                print accuracy

    print 'deg, std: ', deg_std

    '''
    bad_channels = get_bad_channels_polyfit(fourth_band, 4.5, 1)

    _degree = 5
    plt.plot(fourth_band)
    coefs = np.polyfit(np.arange(len(fourth_band)),
                          fourth_band,
                          _degree)
    fitted_line = OrderedDict([])
    for i, dat in enumerate(fourth_band):
        x = [i ** d for d in range(_degree+1)]

        fitted_line[i] = np.sum(coefs * x[::-1])

    plt.plot(fitted_line.values(), color='blue')


    for (i, val), msk in zip(enumerate(fourth_band), fourth_mask):
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
        if i in bad_channels:
            if not bc:
                plt.scatter(i, val, color='yellow', label='Labeled as bad channel')
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
    #plt.savefig('Polyfit_real_peak_finder_2.0_5.png')
    plt.show()
