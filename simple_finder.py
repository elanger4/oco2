import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def get_bad_samples(sample, k, num_std):

    bad_samples = []
    
    for i, dat in enumerate(sample):

        # Select k values before and after specified point
        # and calculate Z-score
        sub_sample = sample[max(0,i-k):min(i+k,len(sample))]
        if np.abs((dat - np.mean(sub_sample))) / np.std(sub_sample) > num_std:
            bad_samples.append(i)

    return bad_samples

if __name__ == "__main__":

    num_samples = 300
    outliers_frac = 0.05

    np.random.seed(45)

    sample = np.zeros(num_samples)
    probs = np.random.uniform(0,1,300)

    bad_true = []


    for i, p in zip(np.arange(num_samples), probs):
        if p < outliers_frac:
            # Outlier
            sample[i] = np.random.normal(int(i/50),5)
            bad_true.append(i)
        else:
            # Good sample
            sample[i] = np.random.normal(int(i/50),1)
        

    bad_pred = get_bad_samples(sample, 10, 2.5)

    bad_true_full = np.zeros(num_samples)
    bad_pred_full = np.zeros(num_samples)

    for i in bad_true:
        bad_true_full[i] = 1

    for i in bad_pred:
        bad_pred_full[i] = 1
    
    '''
    k_std = (0,0)
    accuracy = 0
    for k in range((num_samples/2)-1):
        for std in np.arange(2,5, 0.25):
            bad_pred = get_bad_samples(sample, k, std)

            bad_true_full = np.zeros(num_samples)
            bad_pred_full = np.zeros(num_samples)

            for i in bad_true:
                bad_true_full[i] = 1

            for i in bad_pred:
                bad_pred_full[i] = 1

            results = confusion_matrix(bad_true_full, bad_pred_full)
            temp_accuracy = float(results[0][0] + results[1,1]) / num_samples
            if temp_accuracy > accuracy:
                accuracy = temp_accuracy
                k_std = (k, std)

    print 'Accuracy: ', accuracy
    print 'k, std: ', k_std
    '''

    for i in bad_true:
        bad_true_full[i] = 1

    for i in bad_pred:
        bad_pred_full[i] = 1

    #results = confusion_matrix(bad_true_full, bad_pred_full)
    plt.plot(sample, color='blue')

    good = False
    good_as_bad = False
    missed = False

    for i, s in enumerate(sample):
        # Good sample classified as bad
        if i not in bad_true and i in bad_pred:
            if not good:
                plt.scatter(i, s, color='red', label='Type I Error (marked good sample as bad)')
                good = True
            else:
                plt.scatter(i, s, color='red')

        # Correctly caught bad sample
        if i in bad_true and i in bad_pred:
            if not good_as_bad:
                plt.scatter(i, s, color='green', label='Correctly caught bad sample')
                good_as_bad = True
            else:
                plt.scatter(i, s, color='green')

        # Did not catch bad sample
        if i in bad_true and i not in bad_pred:
            if not missed:
                plt.scatter(i, s, color='orange', label='Type II Error(did not catch bad sample)')
                missed = True
            else:
                plt.scatter(i, s, color='orange')

    plt.legend()
    plt.savefig('simple_peak_finder.png')
    plt.show()
