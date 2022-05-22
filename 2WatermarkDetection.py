# author:   Lena Luisa Feiler
# ID:       i6246119

import cv2
import numpy as np
import dctMethods as dct
import math

threshold = 0.1


def has_watermark(img, omega, orig_img):
    dct_img = dct.blockwise_dct(img)
    k_mask = dct.k_largest_values_blockwise(dct_img)

    orig_img_dct = dct.blockwise_dct(orig_img)
    k_mask_original = dct.k_largest_values_blockwise(orig_img_dct)

    omega_estimate = estimate_watermark(k_mask, k_mask_original)
    omega_mean = omega.mean()
    mean_omega_estimate = omega_estimate.mean()

    # the image does not contain a watermark since both the mystery and the original image are exactly the same
    if mean_omega_estimate == 0:
        return False

    gamma = get_gamma(omega_estimate, mean_omega_estimate, omega, omega_mean)
    print(gamma)

    if gamma < threshold:
        return False
    else:
        return True


# returns the mean of the estimated watermark
def estimate_watermark(k_mask, image):
    sum_omega = np.array([0]*dct.k)
    img_size = k_mask.shape
    # iterate through each block
    for i in np.r_[:img_size[0]:dct.block_size]:
        for j in np.r_[:img_size[1]:dct.block_size]:
            block = k_mask[i:i + dct.block_size, j:j + dct.block_size]
            img_block = image[i:i + dct.block_size, j:j + dct.block_size]
            sum_omega = np.add(sum_omega, estimate_omega(block, img_block))

    return sum_omega


def estimate_omega(block, img_block):
    omega_est = []
    #myst_ph = block[0][0]  # keep for later (DC coefficient)
    #myst_ph[0][0] = 0  # will not be one of the k largest magnitude values

    ################ check if this is correct with absolut
    one_dim_myst = np.reshape(block, (-1))
    k_mystery = (abs(one_dim_myst).argsort())[::-1]

    #ph = img_block[0][0]  # keep for later (DC coefficient)
    #ph[0][0] = 0  # will not be one of the k largest magnitude values

    one_dim = np.reshape(img_block, (-1))
    k_original = (abs(one_dim).argsort())[::-1]

    for i in range(dct.k):
        # ci = coefficient of the image
        # ci_hat coefficient of the mystery image

        ci = one_dim[k_original[i]]
        ci_hat = one_dim_myst[k_mystery[i]]
        wi = (ci_hat - ci) / (dct.alpha * ci)  # wi_hat = (ci_hat - ci) / alpha * ci

        omega_est.append(wi)

    return omega_est


def get_gamma(omega_est, o_mean_est, omega, o_mean):
    num = 0
    den_est = 0
    den_omega = 0
    for i in range(dct.k):
        num += ((omega_est[i] - o_mean_est) * (omega[i] - o_mean))
        den_est = (omega_est[i] - o_mean_est)**2
        den_omega = (omega[i] - o_mean)**2

    return num / (math.sqrt(den_est * den_omega))


# read in picture in gray scale
img_original = cv2.imread("iivp/pictures/elephants.jpg", 0)
# resize image to make it smaller (about half its size) and also the size to be divisible by the blocksize
new_width = math.floor(int(img_original.shape[1] / 2)/dct.block_size)*dct.block_size
new_height = math.floor(int(img_original.shape[0] / 2)/dct.block_size)*dct.block_size
orig_grey = cv2.resize(img_original, (new_width, new_height))  # resize image

# read in picture in gray scale
img_wm = cv2.imread("iivp/resultPictures/exercise2/watermark_inv.jpg", 0)

####### to do change to real omega
omega_original = np.array([0,   255,   0,   0,   0, 0,   0,   0,  0,  0])
wm_watermarked = has_watermark(img_wm, omega_original, orig_grey)
print(wm_watermarked)
orig_watermarked = has_watermark(orig_grey, omega_original, orig_grey)
print(orig_watermarked)


