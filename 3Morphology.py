# author:   Lena Luisa Feiler
# ID:       i6246119

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


def count_oranges():
    return None


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


# erosion followed by dilation
def perform_opening(img, size):
    ero = cv2.erode(img, get_struc_element_circle(size))
    dil = cv2.dilate(ero, get_struc_element_circle(size))
    return dil


def get_struc_element_circle(n):
    # https://stackoverflow.com/questions/44505504/how-to-make-a-circular-kernel
    center = (n-1)/2
    distances = np.indices((n, n)) - np.array([center, center])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - center) <= 0).astype(np.uint8)
    return kernel


def get_background_element(circle, size):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return rect - circle


# https://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib
def display_img_with_grid(img):
    # define the grid spacing
    dx, dy = 10, 10

    # colour of the grid lines
    grid_color = [0, 0, 0]

    # Modify the image to include the grid
    img[:, ::dy] = grid_color
    img[::dx, :] = grid_color

    # Show the result
    plt.imshow(img)
    plt.show()


def create_bw_img(img):
    img[img > 100] = 255
    img[img <= 100] = 0

    return img


# read in picture
orange_rgb = cv2.imread("iivp/pictures/oranges.jpg")
orange_rgb = cv2.cvtColor(orange_rgb,  cv2.COLOR_BGR2RGB)
tree_rgb = cv2.imread("iivp/pictures/orangetree.jpg")
tree_rgb = cv2.cvtColor(tree_rgb,  cv2.COLOR_BGR2RGB)

orange_grey = cv2.cvtColor(orange_rgb,  cv2.COLOR_RGB2GRAY) #convert to gray scale
tree_grey = cv2.cvtColor(tree_rgb,  cv2.COLOR_RGB2GRAY)

# display the rgb with a grid to measure the size of the orange by eye
#display_img_with_grid(orange_rgb)
#display_img_with_grid(tree_rgb)

# the determined kernel sizes
kernel_size_orange = 160
kernel_size_tree = 160 #190

op_orange = perform_opening(orange_grey, kernel_size_orange)
cv2.imwrite('iivp/resultPictures/exercise3/orange_op.jpg', op_orange)

op_tree = perform_opening(tree_grey, kernel_size_tree)
cv2.imwrite('iivp/resultPictures/exercise3/tree_op.jpg', op_tree)

#
bw_orange = create_bw_img(op_orange)
cv2.imwrite('iivp/resultPictures/exercise3/bw_orange.jpg', bw_orange)

bw_tree = create_bw_img(op_tree)
cv2.imwrite('iivp/resultPictures/exercise3/bw_tree.jpg', bw_tree)




# convert image to black and white:
# https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
#(thresh, bw_orange) = cv2.threshold(orange_grey, math.floor(255/2), 255, cv2.THRESH_BINARY)
#cv2.imwrite('iivp/resultPictures/exercise3/BW_oranges.jpg', bw_orange)
#(thresh, bw_tree) = cv2.threshold(tree_grey, math.floor(255/2), 255, cv2.THRESH_BINARY)
#cv2.imwrite('iivp/resultPictures/exercise3/BW_orangetree.jpg', bw_tree)