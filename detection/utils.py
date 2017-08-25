import cv2
import glob
import numpy as np
from detection import feature_extraction as fe


def get_images(dir, type='png'):
    """Read in all images from a directory of a given type"""
    if type not in ('png', 'jpeg', 'jpg'):
        raise ValueError("Not a supported image file.")

    images = (img for img in glob.glob(dir + "/*." + type))
    return (cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in images)


def draw_boxes(img, bboxes, *, color=(0, 0, 255), thick=6):
    """Draws rectangles for the given bounding boxes coordinates"""
    imcopy = np.copy(img)

    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    return imcopy


def slide_windows(img, *, x_start=None, x_stop=None, y_start=None, y_stop=None,
                  xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Return a list of windows to search over"""
    x, y = img.shape[1], img.shape[0]

    # If start and end points are not specified, use the entire image
    if x_start is None:
        x_start = 0
    if x_stop is None:
        x_stop = x
    if y_start is None:
        y_start = 0
    if y_stop is None:
        y_stop = y

    # Compute span of region
    x_span = x_stop - x_start
    y_span = y_stop - y_start

    # Number of pixels per step along x and y direction
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1- xy_overlap[1]))

    # Compute number of windows in the x and y direction
    nx_buffer = np.int(xy_window[0] * xy_overlap[0])
    ny_buffer = np.int(xy_window[1] * xy_overlap[1])
    nx_windows = np.int((x_span - nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((y_span - ny_buffer)/ny_pix_per_step)

    window_list = []

    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start
            endy = starty + xy_window[1]

            window_list.append(((startx, starty), (endx, endy)))

    return window_list


def search_windows(img, windows, clf, scaler, *, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
                   orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                   spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Given an image and a list of windows to search, the function searches
    over these windows to classify the images and return positive results as a 
    list of window coordinates
    """
    on_windows = []
    for window in windows:
        # Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        features = fe.extract_features(
                            [test_img], cspace=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            hist_range=hist_range,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

        test_features = scaler.transform(np.array(features[0]).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)

    return on_windows


if __name__ == "__main__":
    DIR = "../test_images"
    print(list(get_images(DIR, type='jpg')))
