# author:   Lena Luisa Feiler
# ID:       i6246119
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


def granulometry(img, max_size):
    surfAreas = np.zeros(max_size)
    for i in range(max_size):
        sucOpen = perform_opening(img, i)
        surfAreas[i] = np.sum(sucOpen)  # calculate surface area of the opened image
    return surfAreas


# erosion followed by dilation
def perform_opening(img, size):
    se = get_struc_element_circle(size)
    ero = cv2.erode(img, se)
    dil = cv2.dilate(ero, se)
    return dil


def get_struc_element_circle(rad):
    n = rad*2
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

    plt.plot(x)
    plt.ylabel('Differences in surface area')
    plt.xlabel('disc radius')
    plt.savefig("iivp/resultPictures/exercise3/" + name + ".jpg")
    plt.figure()


# read in picture in greyscale
grey_img = cv2.imread("iivp/pictures/jar.jpg", 0)
# perform granulometry
plot_granulometry(granulometry(grey_img, 100), 'oranges')

