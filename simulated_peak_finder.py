import matplotlib.pyplot as plt
import numpy as np
import sys

from sklearn.metrics import confusion_matrix

def get_bad_channels_zscore(channel, k, num_std):

    bad_channels = []
    
    for i, dat in enumerate(channel):

        # Select k values before and after specified point
        # and calculate Z-score
        sub_channel = channel[max(0,i-k):min(i+k+1,len(channel)+1)]
        sub_channel = np.delete(sub_channel, k)
            
        if np.std(sub_channel) == 0:
            print sub_channel
        if np.abs((dat - np.mean(sub_channel))) / np.std(sub_channel) > num_std:
            bad_channels.append(i)

    return bad_channels

def get_bad_channels_polyfit(channel, k, _degree):
    bad_channels = []
    
    for i, dat in enumerate(channel):
        sub_channel = channel[max(0,i-k):min(i+k+1,len(channel))]
        sub_channel = np.delete(sub_channel, k)

        coefs = numpy.polyfit(np.arange(len(sub_channel),
                              sub_channel,
                              _degree))
         
        # TODO: Continue with the idea of fitting a polynomial


if __name__ == "__main__":

    num_channels = 300
    outliers_frac = 0.05

    np.random.seed(42)

    channel = np.zeros(num_channels)
    probs = np.random.uniform(0,1,300)

    bad_true = []


    for i, p in zip(np.arange(num_channels), probs):
        if p < outliers_frac:
            # Outlier
            channel[i] = np.random.normal(int(i/50),5)
            bad_true.append(i)
        else:
            # Good channel
            channel[i] = np.random.normal(int(i/50),1)
        

    bad_pred = get_bad_channels_zscore(channel, 5, 3.75)

    bad_true_full = np.zeros(num_channels)
    bad_pred_full = np.zeros(num_channels)

    '''
    for i in bad_true:
        bad_true_full[i] = 1

    for i in bad_pred:
        bad_pred_full[i] = 1
    
    k_std = (0,0)
    accuracy = 0
    for k in range(1, (num_channels/2)-1):
        for std in np.arange(2,5, 0.25):
            bad_pred = get_bad_channels_zscore(channel, k, std)

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


    #results = confusion_matrix(bad_true_full, bad_pred_full)
    plt.plot(channel, color='blue')

    good = False
    good_as_bad = False
    missed = False

    for i, s in enumerate(channel):
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
    #plt.savefig('simple_peak_finder.png')
    plt.show()
