#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from collections import Counter
from contextlib import closing
#from matplotlib.backends.backend_pdf import PdfPages

#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True
plt.rcParams.update({'font.size': 8})


def find_bad_channels(fp1, fp2, m, b, num_std):

    bad_channels = []

    fitted_line = m * fp1 + b
    
    diff = fp2 - fitted_line

    diff = diff.astype(np.float64)

    diff_mean = np.mean(diff)
    diff_std = np.std(diff)

    for i, d in enumerate(diff):
        if np.abs(d - diff_mean) / diff_std > num_std:
            bad_channels.append(i)
        
    return bad_channels            
        

def o2_footprint_compare(inputfile, outputfile):
    with closing(h5py.File(inputfile, 'r')) as l1:
        o2 = np.array(l1['SoundingMeasurements']['radiance_o2'][0])
    
        a = [2,3,5,6]
        f, axarr = plt.subplots(2,2)
        all_bad = []
        for i, fp in enumerate(a):

            m, b = np.polyfit(o2[4], o2[fp], 1)

            bad_channels = find_bad_channels(o2[4], o2[fp], m, b, 3.5)

            print bad_channels
            all_bad.append(bad_channels)
            bad_values = {}
            for c in bad_channels:
                bad_values[o2[4][c]] = o2[fp][c]
                
            axarr[i/2][i%2].scatter(o2[4], o2[fp], color='black')
            axarr[i/2][i%2].plot(o2[4], m * o2[4] + b, '-', label="Fitted Line")
            axarr[i/2][i%2].scatter(bad_values.keys(), bad_values.values(), label="Marked as bad channels", color='blue')
            axarr[i/2, i%2].set_title('Footprint 4 vs. Footprint ' + str(fp))
            f.subplots_adjust(hspace=0.7, wspace=0.4)
        
        counts = Counter([v for i in all_bad for v in i])
        counts = filter(lambda (k,v) : v > 3, counts.iteritems() )
        counts = [p[0] for p in counts][::-1]
        print counts


        
        common_outliers =  {}
        for i, fp in enumerate(a):
            for c in counts:
                common_outliers[o2[4][c]] = o2[fp][c]
            axarr[i/2][i%2].scatter(common_outliers.keys(), common_outliers.values(), label="Common bad channels", color='red')

        #plt.annotate('Most common channel outliers: ' + str(counts), xy=(0.5, 0.2))
        plt.savefig('adjacent_footprint_outliers_comparison_4_3.5.png')
        #plt.legend()
        plt.show()
        plt.close()
        
        f, axarr = plt.subplots(2,2)
        for i, fp in enumerate(a):
            axarr[i/2][i%2].plot(o2[fp], color='blue')
            axarr[i/2][i%2].set_title('Footprint ' + str(fp))
            #plt.plot(o2[fp], color='blue', label='Footprint 4')
            outliers = [(v, o2[4][v]) for v in counts]

            x = [p[0] for p in outliers]
            y = [p[1] for p in outliers]

            axarr[i/2][i%2].scatter(x, y, color='red', label='Outliers in other footprints')
            #plt.scatter(x, y, color='red', label='Outliers in other footprints')
        #plt.savefig('all_foodprint_4_neighbor_outliers.png')
        plt.show()

def weak_co2_footprint_compare(inputfile, outputfile):
    with closing(h5py.File(inputfile, 'r')) as l1:
        o2 = np.array(l1['SoundingMeasurements']['radiance_weak_co2'][0])
    
        a = [5,7]
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        all_bad = []
        axarr = {0:ax1, 1:ax2}
        for i, fp in enumerate(a):

            m, b = np.polyfit(o2[6], o2[fp], 1)

            bad_channels = find_bad_channels(o2[6], o2[fp], m, b, 3.5)

            print bad_channels
            all_bad.append(bad_channels)
            bad_values = {}
            for c in bad_channels:
                bad_values[o2[6][c]] = o2[fp][c]
                
            axarr[i].scatter(o2[4], o2[fp], color='black')
            axarr[i].plot(o2[4], m * o2[4] + b, '-', label="Fitted Line")
            axarr[i].scatter(bad_values.keys(), bad_values.values(), label="Marked as bad channels", color='blue')
            axarr[i].set_title('Footprint 6 vs. Footprint ' + str(fp))
            f.subplots_adjust(hspace=0.7, wspace=0.4)
        
        counts = set(all_bad[0]).intersection(all_bad[1])
        print counts

        
        common_outliers =  {}
        for i, fp in enumerate(a):
            for c in counts:
                common_outliers[o2[6][c]] = o2[fp][c]
            axarr[i].scatter(common_outliers.keys(), common_outliers.values(), label="Common bad channels", color='red')

        #plt.annotate('Most common channel outliers: ' + str(counts), xy=(0.5, 0.2))
        plt.savefig('weak_co2_adjacent_footprint_outliers_comparison_6_3.5.png')
        #plt.legend()
        plt.show()
        plt.close()
        
        plt.plot(o2[6], color='blue')
        #plt.set_title('Footprint ' + str(fp))
        #plt.plot(o2[fp], color='blue', label='Footprint 4')
        outliers = [(v, o2[6][v]) for v in counts]

        x = [p[0] for p in outliers]
        y = [p[1] for p in outliers]

        plt.scatter(x, y, color='red', label='Outliers in other footprints')
        '''
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        axarr = {0:ax1, 1:ax2}
        for i, fp in enumerate(a):
            axarr[i].plot(o2[fp], color='blue')
            axarr[i].set_title('Footprint ' + str(fp))
            #plt.plot(o2[fp], color='blue', label='Footprint 4')
            outliers = [(v, o2[6][v]) for v in counts]

            x = [p[0] for p in outliers]
            y = [p[1] for p in outliers]

            axarr[i].scatter(x, y, color='red', label='Outliers in other footprints')
            #plt.scatter(x, y, color='red', label='Outliers in other footprints')
        '''
        plt.savefig('weak_co2_foodprint_4_neighbor_outliers.png')
        plt.show()
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('-i', '--inputfile', default='../oco2_L1bScND_13682a_170126_B7302_170127164246.h5')
    parser.add_argument('-o', '--outputfile', default=None)

    args = parser.parse_args()
    #o2_footprint_compare(**vars(args))
    weak_co2_footprint_compare(**vars(args))
