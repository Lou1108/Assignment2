# author:   Lena Luisa Feiler
# ID:       i6246119
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt


def granulometry(img, max_size):
    surfAreas = []
    for i in range(max_size):
        sucOpen = perform_opening(img, i)
        surfAreas.append(get_surface_area(sucOpen))
    return surfAreas


def get_surface_area(opening):
    return np.count_nonzero(opening)


# erosion followed by dilation
def perform_opening(img, size):
    se = get_struc_element_circle(size)
    ero = cv2.erode(img, se)
    dil = cv2.dilate(ero, se)
    return dil


def get_boundary(img, size):
    circle = get_struc_element_circle(size)
    return img - cv2.erode(img, circle)


def hit_or_miss(img, size):
    circle = get_struc_element_circle(size)
    background = get_background_element(circle, size)  # local background
    ero_circle = cv2.erode(img, circle)
    ero_back = cv2.erode(img, background)
    hit_miss = cv2.bitwise_and(ero_circle, ero_back)
    return hit_miss


def get_background_element(circle, size):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return rect - circle


def get_struc_element_circle(n):
    # https://stackoverflow.com/questions/44505504/how-to-make-a-circular-kernel
    center = (n - 1) / 2
    distances = np.indices((n, n)) - np.array([center, center])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - center) <= 0).astype(np.uint8)
    return kernel


def plot_granulometry(surfAreas, name):
    x = []
    # compute difference between consecutive numbers
    for i in range(len(surfAreas) - 1):
        x.append(abs(surfAreas[i] - surfAreas[i + 1]))

    plt.plot(range(len(x)), x)
    plt.ylabel('Differences in surface area')
    plt.xlabel('disc radius')
    plt.savefig("iivp/resultPictures/exercise3/" + name + ".jpg")
    plt.figure()


# read in picture in greyscale
grey_img = cv2.imread("iivp/pictures/granulometry1.jpg", 0)
grey_img = cv2.resize(grey_img, (math.floor(grey_img.shape[1]/2), math.floor(grey_img.shape[0]/2)))

disc = cv2.imread("iivp/pictures/discs.jpeg", 0)
disc = cv2.resize(disc, (675, 531))
opening_example = perform_opening(disc, 100)
cv2.imwrite('iivp/resultPictures/exercise3/op_example.jpg', opening_example)

gran_orange = granulometry(grey_img, 40)
print(gran_orange)
plot_granulometry(gran_orange, 'oranges')


#print(perform_opening(orange_grey, 160))

#print(get_surface_area(perform_opening(orange_grey, 160)))
