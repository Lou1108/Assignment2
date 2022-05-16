# author:   Lena Luisa Feiler
# ID:       i6246119

import math
import cv2
import numpy as np
import scipy.fftpack
from matplotlib import pyplot as plt

block_size = 8
alpha = 0.001
var = 1


def k_largest_values_blockwise(k_mask, k):
    img_size = k_mask.shape
    # iterate through each block
    for i in np.r_[:img_size[0]:block_size]:
        for j in np.r_[:img_size[1]:block_size]:
            block = k_mask[i:i + block_size, j:j + block_size]
            k_mask[i:i + block_size, j:j + block_size] = keep_k_largest(block, k)
    return k_mask


############# check if this should also be done for the dc coefficient
def keep_k_largest(block, k):
    placeholder = block[0][0]  # keep for later (DC coefficient)
    # block[0][0] = 0  # will not be one of the k largest magnitude values
    one_dim_dct = np.reshape(block, (-1))

    high_indices = (abs(one_dim_dct).argsort())[::-1]
    k_highest_val = np.zeros(one_dim_dct.shape)
    for i in range(k):
        k_highest_val[high_indices[i]] = one_dim_dct[high_indices[i]]

    k_highest_val = k_highest_val.reshape(block.shape)
    #k_highest_val[0][0] = placeholder

    return k_highest_val


def watermark_blockwise(k_mask, k, omega):
    img_size = k_mask.shape
    # iterate through each block
    for i in np.r_[:img_size[0]:block_size]:
        for j in np.r_[:img_size[1]:block_size]:
            block = k_mask[i:i + block_size, j:j + block_size]
            k_mask[i:i + block_size, j:j + block_size] = create_watermark(block, k, omega)
    return k_mask


##### change to blockwise ....
def create_watermark(block, k, omega):
    placeholder = block[0][0]  # keep for later (DC coefficient)
    block[0][0] = 0  # will not be one of the k largest magnitude values
    one_dim_dct = np.reshape(block, (-1))
    high_indices = (one_dim_dct.argsort())[::-1]

    for i in range(k):
        # ci' = c*(1#alpha*omega_i)
        one_dim_dct[high_indices[i]] = one_dim_dct[high_indices[i]] * (1 + alpha * omega[i])

    one_dim_dct = one_dim_dct.reshape(block.shape)
    one_dim_dct[0][0] = placeholder

    return one_dim_dct


def create_watermark_non_dc(k):
    return np.random.normal(0, var, k).astype(np.uint8)  # Gaussian Noise


# performing 2d dct on an image in a block-wise manner
def blockwise_dct(img):
    img_size = img.shape
    dct = np.zeros(img.shape)

    for i in np.r_[:img_size[0]:block_size]:
        for j in np.r_[:img_size[1]:block_size]:
            dct[i:i + block_size, j:j + block_size] = dct_2d(img[i:i + block_size, j:j + block_size])

    return dct


def blockwise_idct(img):

    img_size = img.shape
    idct = np.zeros(img_size)

    for i in np.r_[:img_size[0]:block_size]:
        for j in np.r_[:img_size[1]:block_size]:
            idct[i:(i + block_size), j:(j + block_size)] = idct_2d(img[i:(i + block_size), j:(j + block_size)])
    return idct


############################ why ortho?
def dct_2d(img):
    # calculates the 2d dct and performs orthogonalization
    return scipy.fftpack.dct(scipy.fftpack.dct(img.T, norm='ortho').T, norm='ortho')


def idct_2d(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a.T, norm='ortho').T, norm='ortho')


# read in picture in gray scale
img_grey = cv2.imread("iivp/pictures/elephants.jpg", 0)
# resize image to make it smaller (about half its size) and also the size to be divisible by the blocksize
new_width = math.floor(int(img_grey.shape[1] / 2)/block_size)*block_size
new_height = math.floor(int(img_grey.shape[0] / 2)/block_size)*block_size
img_grey = cv2.resize(img_grey, (new_width, new_height))  # resize image

##################### exercise 1.1 ########################
# transforming image to dct domain
dct_img = blockwise_dct(img_grey)
cv2.imwrite('iivp/resultPictures/exercise2/DCT_image.jpg', dct_img)
# save image with an adjusted scale to show the blocks more clearly
plt.figure()
plt.imsave('iivp/resultPictures/exercise2/DCT_image_threshold.jpg', dct_img, vmax=np.max(dct_img) * 0.01, vmin=0)

dct_inv = blockwise_idct(dct_img)
cv2.imwrite('iivp/resultPictures/exercise2/k_mask_10000_inv.jpg', dct_inv)

# finding a good value for k
k = 30
print("image ", dct_img.shape)
dct_larger_k = k_largest_values_blockwise(dct_img, k)
cv2.imwrite('iivp/resultPictures/exercise2/k_mask_10000.jpg', dct_larger_k)

dct_k_inv = blockwise_idct(dct_larger_k)
cv2.imwrite('iivp/resultPictures/exercise2/k_mask_10000_inv.jpg', dct_k_inv)

omega = create_watermark_non_dc(k)
## adding a watermark
dct_watermarked = watermark_blockwise(dct_larger_k, k, omega)
cv2.imwrite('iivp/resultPictures/exercise2/watermark.jpg', dct_watermarked)

dct_watermark_inv = blockwise_idct(dct_watermarked)
cv2.imwrite('iivp/resultPictures/exercise2/watermark_inv.jpg', dct_watermark_inv)

# difference image
difference_img = cv2.subtract(dct_inv, dct_watermark_inv)
cv2.imwrite('iivp/resultPictures/exercise2/DifferenceImage.jpg', difference_img)

# histogram of the difference image
plt.hist(difference_img.ravel(), 256, [0, 255])  # histogram for high contrast picture
plt.title('histogram difference image')
plt.savefig('iivp/resultPictures/exercise2/DifferenceImageHist.jpg')
plt.figure()

