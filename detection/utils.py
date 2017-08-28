import cv2
import glob
import numpy as np
from moviepy.editor import VideoFileClip
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from detection import feature_extraction as fe


CLF = joblib.load("../data/svm_model.pkl")
SCALER = joblib.load("../data/scaler.pkl")


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


def search_windows(img, windows, clf, scaler, *, c_space='RGB',
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
                            [test_img], c_space=c_space,
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


def add_heat(heatmap, bbox_list):
    for box in bbox_list:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    return heatmap


def apply_threshold(heatmap, threshold=0):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return label(heatmap), heatmap


def draw_labeled_boxes(img, labels):
    for car_num in range(1, labels[1]+1):
        # Find pixels with each car_num label value
        nonzero = (labels[0] == car_num).nonzero()

        # Identify x and y values of those pixels
        non_zero_y = np.array(nonzero[0])
        non_zero_x = np.array(nonzero[1])

        bbox = ((np.min(non_zero_x), np.min(non_zero_y)), (np.max(non_zero_x), np.max(non_zero_y)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img


from detection import WindowTracker
WINDOW_TRACKER = WindowTracker.WindowTracker(list(get_images("../data/test_images", 'jpg'))[0], 5)


def process_image(image):
    global WINDOW_TRACKER

    # Modify scale values and start stop ranges to search with different windows
    scales = [1.5]
    start_stop_ranges = [(400, 650)]
    total_windows_list = []
    for scale, ranges in zip(scales, start_stop_ranges):
        out_img, windows_list = fe.find_cars(image, CLF, SCALER, y_start=ranges[0], y_stop=ranges[1],
                                             scale=scale, c_space='YCrCb', cells_per_step=1)
        total_windows_list.extend(windows_list)

    WINDOW_TRACKER.add_windows(total_windows_list)

    labels, heat_map = WINDOW_TRACKER.get_windows()
    draw_img = draw_labeled_boxes(np.copy(image), labels)
    return draw_img


def generate_processed_video(video):
    video_output = "../project_output.mp4"
    video_input = VideoFileClip(video)

    processed_video = video_input.fl_image(process_image)
    processed_video.write_videofile(video_output, audio=False)


if __name__ == "__main__":
    DIR = "../test_images"
    print(list(get_images(DIR, type='jpg')))
