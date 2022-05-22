# author:   Lena Luisa Feiler
# ID:       i6246119

import numpy as np
import scipy.fftpack

block_size = 8
alpha = 0.1
var = 0.1 #1
k = 10


def k_largest_values_blockwise(k_mask):
    img_size = k_mask.shape
    # iterate through each block
    for i in np.r_[:img_size[0]:block_size]:
        for j in np.r_[:img_size[1]:block_size]:
            block = k_mask[i:i + block_size, j:j + block_size]
            k_mask[i:i + block_size, j:j + block_size] = keep_k_largest(block)
    return k_mask


############# check if this should also be done for the dc coefficient
def keep_k_largest(block):
    placeholder = block[0][0]  # keep for later (DC coefficient)
    block[0][0] = 0  # will not be one of the k largest magnitude values
    one_dim_dct = np.reshape(block, (-1))

    high_indices = (abs(one_dim_dct).argsort())[::-1]
    k_highest_val = np.zeros(one_dim_dct.shape)
    for i in range(k):
        k_highest_val[high_indices[i]] = one_dim_dct[high_indices[i]]

    k_highest_val = k_highest_val.reshape(block.shape)
    k_highest_val[0][0] = placeholder

    return k_highest_val


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