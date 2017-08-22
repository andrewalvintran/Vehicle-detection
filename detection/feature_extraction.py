import numpy as np
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    hog_args = {"orientations": orient,
                "pixels_per_cell": (pix_per_cell, pix_per_cell),
                "cells_per_block": (cell_per_block, cell_per_block),
                "visualise": vis,
                "feature_vector": feature_vec}
    if vis:
        features, hog_image = hog(img, **hog_args)
        return features, hog_image
    else:
        features = hog(img, **hog_args)
        return features


def color_hist(img, nbins=32, bins_range=(0, 256)):
    """Computes the color histogram features
    For each channel a histogram is computed and concatenated into a single
    feature vector
    """
    num_channels = 3
    channels_hist = (np.histogram(img[:, :, i], bins=nbins, range=bins_range) for i in range(num_channels))
    channel1_hist, channel2_hist, channel3_hist = channels_hist

    # Generating bin centers
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features
