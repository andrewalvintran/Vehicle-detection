import cv2
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


def bin_spatial(img, size=(32, 32)):
    """Computes binned color features"""
    features = cv2.resize(img, size).ravel()
    return features


def extract_features(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):
    """Given a list of images, get the spatial and histogram features, scale them
    and return as a single feature vector
    """
    features = []

    color_conversion = {"HSV": cv2.COLOR_RGB2HSV,
                      "HLS": cv2.COLOR_RGB2HLS,
                      "YUV": cv2.COLOR_RGB2YUV,
                      "LUV": cv2.COLOR_RGB2LUV}
    for img in imgs:
        if cspace != 'RGB':
            feature_image = cv2.cvtColor(img, color_conversion[cspace])
        else:
            feature_image = np.copy(img)

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        features.append(np.concatenate((spatial_features, hist_features)))
    return features
