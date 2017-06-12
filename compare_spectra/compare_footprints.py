
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import sys

from matplotlib.backends.backend_pdf import PdfPages

from collections import Counter, defaultdict
from contextlib import closing

def plot_full_sample(data, masked, predicted):
    f = plt.figure()

    for i, footp in enumerate(data):
        footp = np.array(footp)
        bad = np.where(mask[i] == 1.0)[0]
        bad_vals = footp[bad]

        
        plt.plot(footp)
        plt.scatter(bad, bad_vals, color='red')

        if predicted[i]:
            pred = np.array(predicted[i])
            pred_vals = footp[pred]
            for p, pv in zip(pred, pred_vals):
                # Found
                if pv in bad_vals:
                    print 'FOUND'
                    plt.scatter(p, pv, color='green')
                # Missed
                else:
                    plt.scatter(p, pv, color='blue')

    with PdfPages('all_spectra_with_masked.pdf') as pdf:
        pdf.savefig(f)
    plt.show()

with closing(h5py.File('../oco2_L1bScND_13682a_170126_B7302_170127164246.h5', 'r')) as l1:
    mask = np.array(l1['InstrumentHeader']['snr_coef'][:, :, :, 2][1])

    footprints = range(8)
    
    bad_fp_chan = {fp: {} for fp in footprints}
    #for s in range(100):
    for s in range(100):
        print 'Examining timestep', s
        data = np.array(l1['SoundingMeasurements']['radiance_weak_co2'][s])

        for chan in range(len(data[footprints[0]])):
            for fp in footprints:
                chan_values = [ data[i][chan] for i in footprints if i != fp ]

                if (data[fp][chan] - np.mean(chan_values)) / np.std(chan_values) > 4.5:
                    if fp in bad_fp_chan.keys():
                        if chan in bad_fp_chan[fp].keys():
                            bad_fp_chan[fp][chan] += 1
                        else:
                            bad_fp_chan[fp][chan] = 1
                    else:
                        bad_fp_chan[fp][chan] = 2

    # Discards smallers values
    for fp, chan in bad_fp_chan.iteritems():
        for i, v in chan.iteritems():
            if v < 10:
                bad_fp_chan[fp][i] = 1
        
    f, axarr = plt.subplots(2,4)
    bad_chan_dict = {}
    for i, fp in enumerate(footprints):
        # Compare to known bad pixels
        bad_pixels = np.where(mask[i] == 1.0)[0]
        print 'bad_pixels:', bad_pixels

        bad_chans = []
        for i, chan in bad_fp_chan[fp].iteritems():
            if chan > 1:
                bad_chans.append(i)
        print bad_chans

        bad_chan_dict[fp] = bad_chans
        counts = bad_fp_chan[fp].values()

        correct   = np.intersect1d(bad_pixels, bad_chans)
        missed    = np.setdiff1d(bad_pixels, bad_chans)
        incorrect = np.setdiff1d(bad_chans, bad_pixels)

        print 'Caught', len(correct), ' of ', len(bad_pixels)
        print 'Missed', len(missed), ' of ', len(bad_pixels)
        print 'Misclassifed', len(incorrect)

        colors = ['blue'] * len(counts)
        for i, b in enumerate(bad_chans):
            colors[i] = 'red'

        # Plot things
        axarr[fp/4][fp%4].bar(bad_fp_chan[fp].keys(),counts, color=colors)
        axarr[fp/4][fp%4].set_title('Footprint ' + str(fp))
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
    with PdfPages('footprints_histogram.pdf') as pdf:
        pdf.savefig(f)
    #plt.show()
    plt.close()

    print bad_chan_dict
    plot_full_sample(np.array(l1['SoundingMeasurements']['radiance_weak_co2'][2]), mask, bad_chan_dict)
