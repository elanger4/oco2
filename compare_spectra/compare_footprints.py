
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from matplotlib.backends.backend_pdf import PdfPages

from collections import Counter, defaultdict
from contextlib import closing

with closing(h5py.File('../oco2_L1bScND_13682a_170126_B7302_170127164246.h5', 'r')) as l1:
    full_data = np.array(l1['SoundingMeasurements']['radiance_weak_co2'])

    footprints = range(8)
    
    #bad_fp_chan = {fp: { chan : 0 for chan in range(len(full_data[footprints[0]])) } for fp in footprints}
    bad_fp_chan = {fp: {} for fp in footprints}
    for s in range(100):
        print 'Examining timestep', s
        data = np.array(l1['SoundingMeasurements']['radiance_weak_co2'][s])
        for chan in range(len(data[footprints[0]])):
            for fp in footprints:
                chan_values = [ data[i][chan] for i in footprints if i != fp ]

                if (data[fp][chan] - np.mean(chan_values)) / np.std(chan_values) > 4.0:
                    if fp in bad_fp_chan.keys():
                        if chan in bad_fp_chan[fp].keys():
                            bad_fp_chan[fp][chan] += 1
                        else:
                            bad_fp_chan[fp][chan] = 1
                    else:
                        bad_fp_chan[fp][chan] = 1

    print bad_fp_chan
    for fp, chan in bad_fp_chan.iteritems():
        for i, v in chan.iteritems():
            if v < 10:
                bad_fp_chan[fp][i] = 0
        
    f, axarr = plt.subplots(2,4)
    for fp in footprints:
        counts = bad_fp_chan[fp].values()
        print bad_fp_chan[fp].values()
        axarr[fp/4][fp%4].bar(bad_fp_chan[fp].keys(),counts)
        axarr[fp/4][fp%4].set_title('Footprint ' + str(fp))
        
    with PdfPages('footprints_histogram.pdf') as pdf:
        pdf.savefig(f)
    plt.show()
    '''
    f, axarr = plt.subplots(2,4)
    for fp in footprints:
        axarr[fp/4][fp%4].plot(data[fp])
        axarr[fp/4][fp%4].set_title('Footprint ' + str(fp))
        axarr[fp/4][fp%4].xaxis.set_ticks([])
        axarr[fp/4][fp%4].yaxis.set_ticks([])
        
        if fp in bad_fp_chan.keys():
            for c in bad_fp_chan[fp]:
                axarr[fp/4][fp%4].scatter(c, data[fp][c], color='red')

    plt.show()
    plt.close()

    for fp in footprints:
        plt.plot(data[fp])

        if fp in bad_fp_chan.keys():
            for c in bad_fp_chan[fp]:
                plt.scatter(c, data[fp][c], color='red')
        
    plt.show()
    plt.close()
    '''
