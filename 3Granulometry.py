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
    return sum(sum(opening))


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
grey_img = cv2.resize(grey_img, (math.floor(grey_img.shape[1]/4), math.floor(grey_img.shape[0]/4)))


gran_orange = granulometry(grey_img, 100)
plot_granulometry(gran_orange, 'oranges')

#check_gran(grey_img)

#print(perform_opening(orange_grey, 160))

#print(get_surface_area(perform_opening(orange_grey, 160)))
