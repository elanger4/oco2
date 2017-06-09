import h5py
import math
import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict
from contextlib import closing
from keras.models import Sequential
from keras.layers import LSTM, Dense 

with closing(h5py.File('../oco2_L1bScND_13682a_170126_B7302_170127164246.h5', 'r')) as l1:
    data = np.array(l1['SoundingMeasurements']['radiance_weak_co2'])

    # Naive way of initializing OrderedDict, would like to be more graceful
    channels = OrderedDict([])

    for i in range(len(data[0][0])):
        channels[i] = []

    for sample in data:
        for footprint in sample:
            for i, chan in enumerate(channels):
                channels[i].append(chan)

    channels_train_x = {}
    channels_train_y = {}
    channels_test_x = {}
    channels_test_y = {}

    results = {}

    for k,v in channels.iteritems():
        train = v[:int(len(v)*0.8)]
        test = v[int(len(v)*0.8):]
        channels_train_x[k] = train[:-1]
        channels_train_y[k] = train[1:]
        channels_test_x[k]  = test[:-1]
        channels_test_y[k]  = test[1:]


        model = Sequential()
        model.add(LSTM(4, input_shape=(1, 50)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(channels_train_x[k], channels_train_y[k], nb_epoch=100, batch_size=1, verbose=2) 

        train_predict = model.predict([channels_train_x[k]])
        test_predict = model.predict([channels_test_x[k]])

        trainScore = math.sqrt(mean_squared_error(channels_train_y[k][0], train_predict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(channels_test_y[k][0], test_predict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
    
