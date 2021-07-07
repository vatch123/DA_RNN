import matplotlib.pyplot as plt

from data import read_data

import numpy as np
import pandas as pd


def data_vis():
    """
    Function to visualize the various time series data in our dataset.
    """
    dataroot = 'solar_data.txt'
    debug = False   
    diff = False
    X, y = read_data(dataroot, debug, diff)

    # First plot the original timeseries
    fig = plt.figure(figsize=(40,40))

    fig.add_subplot(3,3,1)
    plt.plot(y)
    plt.title('Avg Global PSP (vent/cor) [W/m^2]')
    # plt.show()

    fig.add_subplot(3,3,2)
    plt.plot(X[:,0])
    plt.title('Avg Zenith Angle [degrees]')
    # plt.show()

    fig.add_subplot(3,3,3)
    plt.plot(X[:,1])
    plt.title('Avg Azimuth Angle [degrees]')
    # plt.show()

    fig.add_subplot(3,3,4)
    plt.plot(X[:,2])
    plt.title('Avg Tower Dry Bulb Temp [deg C]')
    # plt.show()

    fig.add_subplot(3,3,5)
    plt.plot(X[:,3])
    plt.title('Avg Tower RH [%]')
    # plt.show()

    fig.add_subplot(3,3,6)
    plt.plot(X[:,4])
    plt.title('Avg Total Cloud Cover [%]')
    # plt.show()

    fig.add_subplot(3,3,7)
    plt.plot(X[:,5])
    plt.title('Avg Avg Wind Speed @ 6ft [m/s]')
    # plt.show()

    ##########################################################################################
    # Plotting the Fourier Transform of the signals

    freq = np.fft.fftfreq(len(y), 1*60*60)

    fig = plt.figure(figsize=(40,40))

    fig.add_subplot(3,3,1)
    plt.plot(freq, np.abs(np.fft.fft(y)))
    plt.title('Avg Global PSP (vent/cor) [W/m^2]')
    # plt.show()

    fig.add_subplot(3,3,2)
    plt.plot(freq, np.abs(np.fft.fft(X[:,0])))
    plt.title('Avg Zenith Angle [degrees]')
    # plt.show()

    fig.add_subplot(3,3,3)
    plt.plot(freq, np.abs(np.fft.fft(X[:,1])))
    plt.title('Avg Azimuth Angle [degrees]')
    # plt.show()

    fig.add_subplot(3,3,4)
    plt.plot(freq, np.abs(np.fft.fft(X[:,2])))
    plt.title('Avg Tower Dry Bulb Temp [deg C]')
    # plt.show()

    fig.add_subplot(3,3,5)
    plt.plot(freq, np.abs(np.fft.fft(X[:,3])))
    plt.title('Avg Tower RH [%]')
    # plt.show()

    fig.add_subplot(3,3,6)
    plt.plot(freq, np.abs(np.fft.fft(X[:,4])))
    plt.title('Avg Total Cloud Cover [%]')
    # plt.show()

    fig.add_subplot(3,3,7)
    plt.plot(freq, np.abs(np.fft.fft(X[:,5])))
    plt.title('Avg Avg Wind Speed @ 6ft [m/s]')
    # plt.show()

    ##################################################################################################
    # Print correlation matrix

    df = pd.DataFrame(np.c_[y, X])
    df.columns = ['Avg Global PSP (vent/cor) [W/m^2]','Avg Zenith Angle [degrees]','Avg Azimuth Angle [degrees]','Avg Tower Dry Bulb Temp [deg C]','Avg Tower RH [%]','Avg Total Cloud Cover [%]','Avg Avg Wind Speed @ 6ft [m/s]']
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=20)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.show()

data_vis()
