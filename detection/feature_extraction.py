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
