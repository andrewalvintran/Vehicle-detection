import cv2
import numpy as np
from skimage.feature import hog


COLOR_CONVERSION = {"HSV": cv2.COLOR_RGB2HSV,
                    "HLS": cv2.COLOR_RGB2HLS,
                    "YUV": cv2.COLOR_RGB2YUV,
                    "LUV": cv2.COLOR_RGB2LUV,
                    "YCrCb": cv2.COLOR_RGB2YCrCb}


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


def extract_features(imgs, *, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True,
                     hist_feat=True, hog_feat=True):
    """Given a list of images, get the features (spatial, hist, and hog), scale them
    and return as a single feature vector
    """
    features = []

    for img in imgs:
        img_features = []
        if cspace != 'RGB':
            feature_image = cv2.cvtColor(img, COLOR_CONVERSION[cspace])
        else:
            feature_image = np.copy(img)

        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            img_features.append(spatial_features)
        if hist_feat:
            hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            img_features.append(hist_features)
        if hog_feat:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel], orient,
                                                         pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))

                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block,
                                                vis=False, feature_vec=True)

            img_features.append(hog_features)

        features.append(np.concatenate(img_features))

    return features


def find_cars(img, clf, x_scaler, *, orient=9, y_start=None, y_stop=None, scale=1,
              pix_per_cell=8, cell_per_block=2, spatial_size=(32, 32), hist_bins=32):
    """Searches for cars within a specific window
    This extracts hog features once and sub samples the hog features for a given frame
    """
    draw_img = np.copy(img)

    img_tosearch = img[y_start:y_stop,:,:]

    if scale != 1:
        imgshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imgshape[1]/scale), np.int(imgshape[0]/scale)))

    channels = [img_tosearch[:,:,i] for i in range(3)]
    ch1, ch2, ch3 = channels

    nx_blocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    ny_blocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2
    nx_steps = (nx_blocks - nblocks_per_window) // cells_per_step
    ny_steps = (ny_blocks - nblocks_per_window) // cells_per_step

    hog_channels = [get_hog_features(channel, orient, pix_per_cell, cell_per_block, feature_vec=False)
                    for channel in channels]
    hog1, hog2, hog3 = hog_channels

    for xb in range(nx_steps):
        for yb in range(ny_steps):
            y_pos = yb*cells_per_step
            x_pos = xb*cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[y_pos:y_pos+nblocks_per_window, x_pos:x_pos+nblocks_per_window].ravel()
            hog_feat2 = hog2[y_pos:y_pos+nblocks_per_window, x_pos:x_pos+nblocks_per_window].ravel()
            hog_feat3 = hog3[y_pos:y_pos+nblocks_per_window, x_pos:x_pos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            x_left = x_pos*pix_per_cell
            y_top = y_pos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(img_tosearch[y_top:y_top+window, x_left:x_left+window], (64,64))

            # Color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = x_scaler.transform(np.hstack(
                (spatial_features, hist_features, hog_features)).reshape(1, -1))

            prediction = clf.predict(test_features)

            if prediction == 1:
                x_box_left = np.int(x_left*scale)
                y_top_draw = np.int(y_top*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (x_box_left, y_top_draw+y_start),
                              (x_box_left+win_draw, y_top_draw+win_draw+y_start), (0,0,255),6)

    return draw_img
