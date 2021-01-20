import sys
sys.path.append("./../")
import math as m
import numpy as np
from SpectralImagesClassification.Utils import cart2sph,pol2cart,augment_EEG
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

def plot_locs(locs):
    plt.plot(locs['x'], locs['y'], '*')
    plt.show()

def gen_images(locs, features, n_gridpoints, normalize=True):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] // nElectrodes
    for c in range(int(n_colors)):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])

    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))

    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i+1, nSamples), end='\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])

    temp_interp = np.swapaxes(np.asarray(temp_interp), 0, 1)
    temp_interp = np.swapaxes(temp_interp, 1, 3)
    return temp_interp    # swap axes to have [samples, colors, W, H]


def createSequencesForLstm(images, lstm_sequence_length, overlapping_sequences = True, stride=1 ):
    """
    Transform list of images into sequences for a LSTM model.
    :param stride:
    :param overlapping_sequences:
    :param lstm_sequence_length:
    :param images:
    :return:
    """
    images_sequences = []
    idx2 = 0
    while idx2 + lstm_sequence_length <= images.shape[0]:
        images_sequences.append(images[idx2:idx2 + lstm_sequence_length])
        if overlapping_sequences:
            idx2 = idx2 + stride
        else:
            idx2 = idx2 + lstm_sequence_length

    images = np.array(images_sequences)

    return images