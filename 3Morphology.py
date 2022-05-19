# author:   Lena Luisa Feiler
# ID:       i6246119

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


def count_oranges(img, kernel_rad):
    total_area = get_total_circle_surface(img)
    area_one_element = np.pi * kernel_rad**2
    return math.floor(total_area/area_one_element)


# erosion followed by dilation
def perform_opening(img, size):
    ero = cv2.erode(img, get_struc_element_circle(size))
    dil = cv2.dilate(ero, get_struc_element_circle(size))
    return dil


def get_struc_element_circle(rad):
    n = rad*2
    # https://stackoverflow.com/questions/44505504/how-to-make-a-circular-kernel
    center = (n - 1) / 2
    distances = np.indices((n, n)) - np.array([center, center])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - center) <= 0).astype(np.uint8)
    return kernel


# https://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib
def display_img_with_grid(img, name):
    # define the grid spacing
    dx, dy = 10, 10

    # colour of the grid lines
    grid_color = [0, 0, 0]

    # Modify the image to include the grid
    img[:, ::dy] = grid_color
    img[::dx, :] = grid_color

    # save the result
    plt.imsave("iivp/resultPictures/exercise3/"+name+".jpg", img)


def get_total_circle_surface(img):
    return np.count_nonzero(img == 255)


# read in picture in greyscale
orange_grey = cv2.imread("iivp/pictures/oranges.jpg", 0)
orange_grey = cv2.resize(orange_grey, (math.floor(orange_grey.shape[1]), math.floor(orange_grey.shape[0])))
tree_grey = cv2.imread("iivp/pictures/orangetree.jpg", 0)
tree_grey = cv2.resize(tree_grey, (math.floor(tree_grey.shape[1]), math.floor(tree_grey.shape[0])))

# convert image to black and white:
# https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
(thresh, bw_orange) = cv2.threshold(orange_grey, math.floor(255/2), 255, cv2.THRESH_BINARY)
cv2.imwrite('iivp/resultPictures/exercise3/BW_oranges.jpg', bw_orange)
(thresh, bw_tree) = cv2.threshold(tree_grey, math.floor(255/2), 255, cv2.THRESH_BINARY)
cv2.imwrite('iivp/resultPictures/exercise3/BW_orangetree.jpg', bw_tree)


# display the rgb with a grid to measure the size of the orange by eye
orange_rgb = cv2.cvtColor(orange_grey, cv2.COLOR_GRAY2RGB)
tree_rgb = cv2.cvtColor(tree_grey, cv2.COLOR_GRAY2RGB)

display_img_with_grid(orange_rgb, "orange_grid")
display_img_with_grid(tree_rgb, "tree_grid")

# the determined kernel sizes
kernel_size_orange = 85  # 150
kernel_size_tree = 82  #160


# perform opening
op_orange = perform_opening(bw_orange, kernel_size_orange)
cv2.imwrite('iivp/resultPictures/exercise3/orange_op.jpg', op_orange)
op_tree = perform_opening(bw_tree, kernel_size_tree)
cv2.imwrite('iivp/resultPictures/exercise3/tree_op.jpg', op_tree)

# do count
count_o = count_oranges(op_orange, kernel_size_orange)
print(count_o)

count_tree = count_oranges(op_tree, kernel_size_tree)
print(count_tree)

