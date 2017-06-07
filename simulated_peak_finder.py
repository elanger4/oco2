import matplotlib.pyplot as plt
import numpy as np
import sys

from collections import OrderedDict
from sklearn.metrics import confusion_matrix

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

if __name__ == "__main__":

    num_channels = 300
    outliers_frac = 0.05

    np.random.seed(45)

    sample = np.zeros(num_channels)
    probs = np.random.uniform(0,1,300)

    bad_true = []


    for i, p in zip(np.arange(num_channels), probs):
        if p < outliers_frac:
            # Outlier
            sample[i] = np.random.normal(int(i/50),5)
            bad_true.append(i)
        else:
            # Good channel
            sample[i] = np.random.normal(int(i/50),1)
        

    #bad_pred = get_bad_channels_polyfit(sample, 2.0, 10)
    
    '''
    deg_std = (0,0)
    accuracy = 0
    for std in np.arange(2,5, 0.25):
        for d in range(1, 10):
            bad_pred = get_bad_channels_polyfit(sample, std, d)

            bad_true_full = np.zeros(num_channels)
            bad_pred_full = np.zeros(num_channels)

            for i in bad_true:
                bad_true_full[i] = 1

            for i in bad_pred:
                bad_pred_full[i] = 1

            results = confusion_matrix(bad_true_full, bad_pred_full)
            temp_accuracy = float(results[0][0] + results[1,1]) / num_channels
            if temp_accuracy > accuracy and bad_pred:
                accuracy = temp_accuracy
                deg_std = (d, std)
                print accuracy

    print 'Accuracy: ', accuracy
    print 'deg, std: ', deg_std

    '''
    bad_pred = get_bad_channels_polyfit(sample, 2.0, 5)

    bad_true_full = np.zeros(num_channels)
    bad_pred_full = np.zeros(num_channels)

    for i in bad_true:
        bad_true_full[i] = 1

    for i in bad_pred:
        bad_pred_full[i] = 1


    #results = confusion_matrix(bad_true_full, bad_pred_full)
    _degree = 2

    plt.plot(sample, color='black')
    coefs = np.polyfit(np.arange(len(sample)),
                          sample,
                          _degree)
    fitted_line = OrderedDict([])
    print coefs
    for i, dat in enumerate(sample):
        x = [i ** d for d in range(_degree+1)]
        print x

        fitted_line[i] = np.sum(coefs * x[::-1])

    plt.plot(fitted_line.values(), color='blue')

    good = False
    good_as_bad = False
    missed = False

    for i, s in enumerate(sample):
        # Good channel classified as bad
        if i not in bad_true and i in bad_pred:
            if not good:
                plt.scatter(i, s, color='red', label='Type I Error (marked good channel as bad)')
                good = True
            else:
                plt.scatter(i, s, color='red')

        # Correctly caught bad channel
        if i in bad_true and i in bad_pred:
            if not good_as_bad:
                plt.scatter(i, s, color='green', label='Correctly caught bad channel')
                good_as_bad = True
            else:
                plt.scatter(i, s, color='green')

        # Did not catch bad channel
        if i in bad_true and i not in bad_pred:
            if not missed:
                plt.scatter(i, s, color='orange', label='Type II Error(did not catch bad channel)')
                missed = True
            else:
                plt.scatter(i, s, color='orange')

    plt.legend()
    plt.savefig('ploy_peak_finder_2.0_5.png')
    plt.show()
