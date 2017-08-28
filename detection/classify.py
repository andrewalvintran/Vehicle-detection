import numpy as np
import os
import time
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from detection import utils
from detection import feature_extraction as fe


CLF_PATH = "../data/svm_model.pkl"
SCALER_PATH = "../data/scaler.pkl"


def create_model(data):
    x_train, y_train = data["train"]
    x_test, y_test = data["test"]

    t = time.time()
    svc = LinearSVC()
    svc.fit(x_train, y_train)
    t2 = time.time()

    print(round(t2 - t, 2), 'seconds to train SVC...')
    print("Test accuracy of SVC = ", svc.score(x_test, y_test))
    joblib.dump(svc, CLF_PATH)
    return svc


def process_data_for_train(car_features, notcar_features):
    """Given a list of feature vectors for the two classes,
    this will generate the y labels and return a dictionary
    containing the train and test data
    """
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    x = np.vstack((car_features, notcar_features)).astype(np.float64)

    x_scaler = StandardScaler().fit(x)
    scaled_x = x_scaler.transform(x)

    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2,
                                                        random_state = rand_state)
    joblib.dump(x_scaler, SCALER_PATH)
    return (x_scaler,
           {"train": [x_train, y_train],
            "test": [x_test, y_test]})


def main():
    vehicle_dir = '../data/vehicles/*/'
    vehicle_imgs = utils.get_images(vehicle_dir)
    cars = list(vehicle_imgs)

    non_vehicle_dir = '../data/non-vehicles/*/'
    non_vehicle_imgs = utils.get_images(non_vehicle_dir)
    notcars = list(non_vehicle_imgs)

    feature_options = {"spatial_size": (32, 32),
                       "hist_bins": 32,
                       "hog_channel": "ALL",
                       "c_space": "YCrCb"}

    if os.path.exists(CLF_PATH):
        clf = joblib.load(CLF_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        car_features = fe.extract_features(cars, **feature_options)
        notcar_features = fe.extract_features(notcars, **feature_options)

        scaler, data = process_data_for_train(car_features, notcar_features)
        clf = create_model(data)

    video = "../project_video.mp4"
    utils.generate_processed_video(video)


if __name__ == "__main__":
    main()
