# author:   Lena Luisa Feiler
# ID:       i6246119

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import r_, mean
import dctMethods as dct

block_size = 8
alpha = 0.1  #0.001
var = 0.1  #1
k = 15  #7
threshold = 0.5  #0.1

##################################### change dolphin ###########################################
# read in picture in gray scale
orig_img = cv2.imread("iivp/pictures/cameraman.tif", 0)
# resize image to make it smaller (about half its size) and also the size to be divisible by the blocksize
#new_width = math.floor(int(orig_img.shape[1] / 2) / block_size) * block_size
#new_height = math.floor(int(orig_img.shape[0] / 2) / block_size) * block_size
#orig_img = cv2.resize(orig_img, (new_width, new_height))  # resize image

###################################################### exercise 1.2 ###################################################
# transforming image to dct domain
dct_img = dct.blockwise_dct(orig_img, block_size)
cv2.imwrite('iivp/resultPictures/exercise2/DCT_image.jpg', dct_img)
# save image with an adjusted scale to show the blocks more clearly
plt.figure()
plt.imsave('iivp/resultPictures/exercise2/DCT_image_threshold.jpg', dct_img, vmax=np.max(dct_img) * 0.01, vmin=0)

dct_inv = dct.blockwise_idct(dct_img, block_size)
cv2.imwrite('iivp/resultPictures/exercise2/dct_inv.jpg', dct_inv)

# finding a good value for k
print("image ", dct_img.shape)
dct_larger_k = dct.k_largest_values_blockwise(dct_img, block_size, k)
cv2.imwrite('iivp/resultPictures/exercise2/k_mask.jpg', dct_larger_k)

dct_k_inv = dct.blockwise_idct(dct_larger_k, block_size)
cv2.imwrite('iivp/resultPictures/exercise2/k_mask_inv.jpg', dct_k_inv)

omega = dct.create_watermark_omega(k, var)
print("omega :" , omega)
## adding a watermark
dct_watermarked = dct.watermark_blockwise(dct_larger_k, k, omega, block_size, alpha)
cv2.imwrite('iivp/resultPictures/exercise2/watermark.jpg', dct_watermarked)

wm_img = dct.blockwise_idct(dct_watermarked, block_size)
cv2.imwrite('iivp/resultPictures/exercise2/watermark_inv.jpg', wm_img)

# difference image
difference_img = cv2.subtract(dct_inv, wm_img) ########################## difference_image = np.abs(img - inverse_water_mark)
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

