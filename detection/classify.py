from sklearn.svm import LinearSVC
from detection import utils


def create_model(x_train, y_train):
    svc = LinearSVC()
    svc.fit(x_train, y_train)
    return svc


def main():
    vehicle_dir = '../data/vehicles/*/'
    vehicle_imgs = utils.get_images(vehicle_dir)

    non_vehicle_dir = '../data/non-vehicles/*/'
    non_vehicle_imgs = utils.get_images(non_vehicle_dir)


if __name__ == "__main__":
    main()