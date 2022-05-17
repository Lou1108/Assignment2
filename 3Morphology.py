# author:   Lena Luisa Feiler
# ID:       i6246119
import math
import cv2







# read in picture in gray scale
orange_grey = cv2.imread("iivp/pictures/oranges.jpg", 0)
tree_grey = cv2.imread("iivp/pictures/orangetree.jpg", 0)
# convert image to black and white:
# https://techtutorialsx.com/2019/04/13/python-opencv-converting-image-to-black-and-white/
(thresh, bw_orange) = cv2.threshold(orange_grey, math.floor(255/2), 255, cv2.THRESH_BINARY)
cv2.imwrite('iivp/resultPictures/exercise3/BW_oranges.jpg', bw_orange)
(thresh, bw_tree) = cv2.threshold(orange_grey, math.floor(255/2), 255, cv2.THRESH_BINARY)
cv2.imwrite('iivp/resultPictures/exercise3/BW_orangetree.jpg', bw_tree)




