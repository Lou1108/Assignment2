# author:   Lena Luisa Feiler
# ID:       i6246119

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import dctMethods as dct

block_size = 8
alpha = 0.15
var = 0.1
k = 9
threshold = 0.9

# read in picture in gray scale
orig_img = cv2.imread("iivp/pictures/dolphin.jpg", 0)
# resize image to make it smaller (about half its size) and also the size to be divisible by the blocksize
new_width = math.floor(int(orig_img.shape[1] / 2) / block_size) * block_size
new_height = math.floor(int(orig_img.shape[0] / 2) / block_size) * block_size
orig_img = cv2.resize(orig_img, (new_width, new_height))  # resize image
cv2.imwrite('iivp/resultPictures/exercise2/dolphin_grey.jpg', orig_img)  # save grey image

###################################################### exercise 1.2 ###################################################
# finding a good value for k
dct_larger_k = dct.k_largest_values_blockwise(orig_img, block_size, 8)
dct_k_inv = dct.blockwise_idct(dct_larger_k, block_size)
cv2.imwrite('iivp/resultPictures/exercise2/k_mask_8.jpg', dct_k_inv)

dct_larger_k = dct.k_largest_values_blockwise(orig_img, block_size, 9)
dct_k_inv = dct.blockwise_idct(dct_larger_k, block_size)
cv2.imwrite('iivp/resultPictures/exercise2/k_mask_9.jpg', dct_k_inv)

# adding a watermark
omega = dct.create_watermark_omega(k, var)  # pseudo random sequence of gaussian noise

dct_watermarked = dct.watermark_blockwise(dct_larger_k, k, omega, block_size, alpha)
cv2.imwrite('iivp/resultPictures/exercise2/watermark.jpg', dct_watermarked)

wm_img = dct.blockwise_idct(dct_watermarked, block_size)
cv2.imwrite('iivp/resultPictures/exercise2/watermark_inv.jpg', wm_img)

# difference image of the original and watermarked image
difference_img = cv2.subtract(orig_img, wm_img.astype(np.uint8))
cv2.imwrite('iivp/resultPictures/exercise2/DifferenceImage.jpg', difference_img)

# histogram of the difference image
plt.hist(difference_img.ravel(), 256, [0, 255])  # histogram for high contrast picture
plt.title('histogram difference image')
plt.savefig('iivp/resultPictures/exercise2/DifferenceImageHist.jpg')
plt.figure()


###################################################### exercise 2.2 ###################################################
# orig_img is the original image (1), wm_img is the watermarked image (2) taken from the previous exercise
orig_watermarked = dct.has_watermark(orig_img, omega, orig_img, block_size, k, alpha, threshold)
print("The first image contains a watermark: ", orig_watermarked)
wm_watermarked = dct.has_watermark(wm_img, omega, orig_img, block_size, k, alpha, threshold)
print("The second image contains a watermark: ", wm_watermarked)

