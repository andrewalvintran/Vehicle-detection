import cv2
import glob


def get_images(dir, type='png'):
    """Read in all images from a directory of a given type"""
    if type not in ('png', 'jpeg', 'jpg'):
        raise ValueError("Not a supported image file.")

    images = (img for img in glob.glob(dir + "/*." + type))
    return (cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in images)


if __name__ == "__main__":
    DIR = "../test_images"
    print(list(get_images(DIR, type='jpg')))
