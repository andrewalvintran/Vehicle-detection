import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from detection import utils
from detection import feature_extraction as fe


def create_model(data):
    x_train, y_train = data["train"]
    x_test, y_test = data["test"]

    t = time.time()
    svc = LinearSVC()
    svc.fit(x_train, y_train)
    t2 = time.time()

    print(round(t2 - t, 2), 'seconds to train SVC...')
    print("Test accuracy of SVC = ", svc.score(x_test, y_test))
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

    return {"train": [x_train, y_train],
            "test": [x_test, y_test]}


def main():
    vehicle_dir = '../data/vehicles/*/'
    vehicle_imgs = utils.get_images(vehicle_dir)

    cars = list(vehicle_imgs)
    non_vehicle_dir = '../data/non-vehicles/*/'
    non_vehicle_imgs = utils.get_images(non_vehicle_dir)
    notcars = list(non_vehicle_imgs)

    car_features = fe.extract_features(cars, cspace='RGB', spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256), hog_feat=False)
    notcar_features = fe.extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
                                       hist_bins=32, hist_range=(0, 256), hog_feat=False)

    data = process_data_for_train(car_features, notcar_features)
    clf = create_model(data)

    car_features = fe.extract_features(cars, hog_channel='ALL')
    notcar_features= fe.extract_features(notcars, hog_channel='ALL')

    data_hog = process_data_for_train(car_features, notcar_features)
    clf_all_features = create_model(data_hog)


if __name__ == "__main__":
    main()
