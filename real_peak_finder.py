import numpy as np
'''
import pandas as pd

data = pd.read_hdf('ocoL1_data.h5')

print data
'''
import h5py
import matplotlib.pyplot as plt
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


with closing(h5py.File('oco2_L1bScND_13682a_170126_B7302_170127164246.h5', 'r')) as l1:
    samples = np.array(l1['SoundingMeasurements']['radiance_o2'])
    mask = np.array(l1['InstrumentHeader']['snr_coef'][:, :, :, 2])
    fourth_band = samples[0][1]
    fourth_mask = mask[0][1]

    num_channels = len(fourth_band)

    p1 = False
    p2 = False
    p4 = False
    bc = False

    bad_channels = get_bad_channels_zscore(fourth_band, 5, 3)
    print bad_channels

    '''
    bad_true_full = np.zeros(len(fourth_band))
    bad_pred_full = np.zeros(len(fourth_band))

    for i in bad_true:
        bad_true_full[i] = 1

    for i in bad_pred:
        bad_pred_full[i] = 1
    
    k_std = (0,0)
    accuracy = 0
    for k in range(1, (num_channels/2)-1):
        for std in np.arange(2,5, 0.25):
            bad_pred = get_bad_channels_zscore(fourth_band, k, std)

            bad_true_full = np.zeros(num_channels)
            bad_pred_full = np.zeros(num_channels)

            for i in bad_true:
                bad_true_full[i] = 1

            for i in bad_pred:
                bad_pred_full[i] = 1

            results = confusion_matrix(bad_true_full, bad_pred_full)
            temp_accuracy = float(results[0][0] + results[1,1]) / num_channels
            if temp_accuracy > accuracy:
                accuracy = temp_accuracy
                k_std = (k, std)

    print 'Accuracy: ', accuracy
    print 'k, std: ', k_std
    '''

    plt.plot(fourth_band)


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
                plt.scatter(i, val, color='red', label='Labeled as bad channel')
                bc = True
            else:
                plt.scatter(i, val, color='red')


    plt.legend()
    plt.savefig('Simple_peak_finder_0_1.png')
    plt.show()
