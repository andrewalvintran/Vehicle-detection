import glob
import matplotlib.image as mpimg


def get_images(dir, type='png'):
    """Read in all images from a directory of a given type"""
    if type not in ('png', 'jpeg', 'jpg'):
        raise ValueError("Not a supported image file.")

    images = (img for img in glob.glob(dir + "/*." + type))
    return (mpimg.imread(img) for img in images)


if __name__ == "__main__":
    DIR = "../test_images"
    print(list(get_images(DIR, type='jpg')))