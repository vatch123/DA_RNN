import numpy as np
import pandas as pd


def read_data(input_path, debug=False, diff=False):
    """Read data.

    Args:
        input_path (str): directory to nasdaq dataset.

    Returns:
        X (np.ndarray): features.
        y (np.ndarray): ground truth.

    """
    df = pd.read_csv(input_path, nrows=150 if debug else None)

    # X = df.iloc[:, 0:-1].values
    X = df.drop(['DATE (MM/DD/YYYY)', 'HOUR-MST', 'Avg Global PSP (vent/cor) [W/m^2]'], axis=1).as_matrix()
    # y = df.iloc[:, -1].values
    y = np.array(df['Avg Global PSP (vent/cor) [W/m^2]'])

    # print(X.shape)

    X[:,4] = np.where(X[:,4]<-10, 0, X[:,4])
    ######################################################################################
    # Stationarize each signal
    if diff:
        y = y[1:]-y[0:-1]
        X = X[1:,:] - X[0:-1,:]

    ##########################################################################################
    # Normalize every data
    y = (y - min(y))
    y /= max(y)

    X = (X - np.min(X, axis=0))
    X /= np.max(X, axis=0)


    return X, y