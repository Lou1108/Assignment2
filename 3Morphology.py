# author:   Lena Luisa Feiler
# ID:       i6246119

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt


# calculates the total number of estimated oranges in the picture based on the radius of the structuring elements
def count_oranges(img, kernel_rad):
    total_area = get_total_circle_surface(img)  # total area that remains after performing the opening
    area_one_element = np.pi * kernel_rad**2  # area of one of the oranges in
    return math.floor(total_area/area_one_element)  # estimated number of elements in the picture


# erosion followed by dilation
def perform_opening(img, size):
    se = get_struc_element_circle(size)  # circular structuring element
    ero = cv2.erode(img, se)
    dil = cv2.dilate(ero, se)  # perform dilation on eroded image
    return dil


def get_struc_element_circle(rad):
    diam = rad*2
    # reference: https://stackoverflow.com/questions/44505504/how-to-make-a-circular-kernel
    center = (diam - 1) / 2  # center of the circle
    distances = np.indices((diam, diam)) - np.array([center, center])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - center) <= 0).astype(np.uint8)
    return kernel


# used to determine the kernel size by inspection
def display_img_with_grid(img, name):
    # reference: https://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib
    dx, dy = 10, 10  # distance between grid lines
    grid_color = [0, 0, 0]  # set grid line colour to black
    # adding the grid lines to the image
    img[:, ::dy] = grid_color
    img[::dx, :] = grid_color
    plt.imsave("iivp/resultPictures/exercise3/"+name+".jpg", img)  # save resulting image


# retrieve total surface area that remains after opening on black and white image
def get_total_circle_surface(img):
    return np.count_nonzero(img == 255)  # count white pixel


# performs successive openings with a step size of 1 and from 0 until the max_size
def granulometry(img, max_size):
    surfAreas = np.zeros(max_size)  # saves the surface areas of the opened images at each opening
    for i in range(max_size):
        sucOpen = perform_opening(img, i)
        surfAreas[i] = np.sum(sucOpen)  # calculate surface area of the opened image
    return surfAreas


# plotting the difference in surface areas
def plot_granulometry(surfAreas, name):
    x = []
    # compute difference between consecutive numbers
    for i in range(len(surfAreas) - 1):
        x.append(abs(surfAreas[i] - surfAreas[i + 1]))

    plt.plot(x)
    plt.ylabel('Differences in surface area')
    plt.xlabel('Disc radius')
    plt.savefig("iivp/resultPictures/exercise3/" + name + ".jpg")
    plt.figure()


###################################################### exercise 3.1 ###################################################
# read in picture as greyscale
orange_grey = cv2.imread("iivp/pictures/oranges.jpg", 0)
tree_grey = cv2.imread("iivp/pictures/orangetree.jpg", 0)

# convert image to black and white:
# https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
(thresh, bw_orange) = cv2.threshold(orange_grey, math.floor(255/2), 255, cv2.THRESH_BINARY)
cv2.imwrite('iivp/resultPictures/exercise3/BW_oranges.jpg', bw_orange)
(thresh, bw_tree) = cv2.threshold(tree_grey, math.floor(255/2), 255, cv2.THRESH_BINARY)
cv2.imwrite('iivp/resultPictures/exercise3/BW_tree.jpg', bw_tree)

# display the rgb with a grid to measure the size of the orange by eye
display_img_with_grid(cv2.cvtColor(orange_grey, cv2.COLOR_GRAY2RGB), "orange_grid")
display_img_with_grid(cv2.cvtColor(tree_grey, cv2.COLOR_GRAY2RGB), "tree_grid")

# the determined kernel sizes
kernel_size_orange = 85
kernel_size_tree = 82

# perform opening
op_orange = perform_opening(bw_orange, kernel_size_orange)
cv2.imwrite('iivp/resultPictures/exercise3/open_orange.jpg', op_orange)
op_tree = perform_opening(bw_tree, kernel_size_tree)
cv2.imwrite('iivp/resultPictures/exercise3/open_tree.jpg', op_tree)

# count the number of elements in the tree
count_o = count_oranges(op_orange, kernel_size_orange)
print('The orange picture contains ', count_o, ' oranges.')

count_tree = count_oranges(op_tree, kernel_size_tree)
print('The orange tree picture contains ', count_tree, ' oranges.')


###################################################### exercise 3.2 ###################################################
# read in picture in greyscale
grey_img = cv2.imread("iivp/pictures/jar.jpg", 0)
grey_img = cv2.resize(grey_img, (math.floor(grey_img.shape[1]/4), math.floor(grey_img.shape[0]/4)))  # resize image

# perform granulometry and plot the results
plot_granulometry(granulometry(grey_img, 100), 'jar_granulometry')
display_img_with_grid(cv2.cvtColor(grey_img, cv2.COLOR_GRAY2RGB), "jar_grid")

