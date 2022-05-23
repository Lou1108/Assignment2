# author:   Lena Luisa Feiler
# ID:       i6246119
import math
import numpy as np
import scipy.fftpack


def k_largest_values_blockwise(k_mask, block_size, k):
    img_size = k_mask.shape
    # iterate through each block
    for i in np.r_[:img_size[0]:block_size]:
        for j in np.r_[:img_size[1]:block_size]:
            block = k_mask[i:i + block_size, j:j + block_size]
            # replaces the block by its watermarked version
            k_mask[i:i + block_size, j:j + block_size] = keep_k_largest(block, k)
    return k_mask


# only keeps the k largest dct coefficients, sets all other dct coefficients to 0
def keep_k_largest(block, k):
    placeholder = block[0][0]  # keep for later (DC coefficient)
    block[0][0] = 0  # will not be one of the k largest magnitude values
    one_dim_dct = np.reshape(block, (-1))

    high_indices = (abs(one_dim_dct).argsort())[::-1]  # sort values by largest magnitudes
    k_highest_val = np.zeros(one_dim_dct.shape)
    for i in range(k):
        k_highest_val[high_indices[i]] = one_dim_dct[high_indices[i]]  # only keeps the k largest coefficients

    k_highest_val = k_highest_val.reshape(block.shape)  # reshape to original block shape
    k_highest_val[0][0] = placeholder  # make sure to not change value of dc coefficient

    return k_highest_val


# performing 2d dct on an image in a block-wise manner
def blockwise_dct(img, block_size):
    img_size = img.shape
    dct = np.zeros(img.shape)

    for i in np.r_[:img_size[0]:block_size]:  # loops through whole image
        for j in np.r_[:img_size[1]:block_size]:
            # performs dct on each block
            dct[i:i + block_size, j:j + block_size] = dct_2d(img[i:i + block_size, j:j + block_size])

    return dct


# performing 2d idct on an image in a block-wise manner
def blockwise_idct(img, block_size):
    img_size = img.shape
    idct = np.zeros(img_size)

    for i in np.r_[:img_size[0]:block_size]:   # loops through whole image
        for j in np.r_[:img_size[1]:block_size]:
            # performs idct on each block
            idct[i:(i + block_size), j:(j + block_size)] = idct_2d(img[i:(i + block_size), j:(j + block_size)])
    return idct


# performing 2d dct on an image block
def dct_2d(block):
    # calculates the 2d dct and performs orthogonalization
    return scipy.fftpack.dct(scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho')


# performing 2d idct on an image block
def idct_2d(block):
    return scipy.fftpack.idct(scipy.fftpack.idct(block.T, norm='ortho').T, norm='ortho')


# inputs watermark in image in a block-wise manner
def watermark_blockwise(k_mask, k, omega, block_size, alpha):
    img_size = k_mask.shape
    # iterate through each block
    for i in np.r_[:img_size[0]:block_size]:
        for j in np.r_[:img_size[1]:block_size]:
            block = k_mask[i:i + block_size, j:j + block_size]
            # replaces the block by its watermarked version
            k_mask[i:i + block_size, j:j + block_size] = create_watermark(block, k, omega, alpha)
    return k_mask


# inserts watermark in a given block in its k coefficients
def create_watermark(block, k, omega, alpha):
    placeholder = block[0][0]  # keep for later (DC coefficient)
    block[0][0] = 0  # will not be one of the k largest magnitude values
    one_dim_dct = np.reshape(block, (-1))
    high_indices = (abs(one_dim_dct).argsort())[::-1]  # sorts values by their largest magnitude

    for i in range(k):
        # ci' = c*(1#alpha*omega_i)
        one_dim_dct[high_indices[i]] = one_dim_dct[high_indices[i]] * (1 + alpha * omega[i])

    one_dim_dct = one_dim_dct.reshape(block.shape)  # reshape to original block shape
    one_dim_dct[0][0] = placeholder  # ensure that the dc coefficient stays its original value

    return one_dim_dct


# creates pseudo random watermarking sequence
def create_watermark_omega(k, var):
    return np.random.normal(0, var, k).astype(np.uint8)  # Gaussian Noise


# determines if a mystery image has a watermark, assuming we know the original and the original omega
def has_watermark(img, omega, orig_img, block_size, k, alpha, threshold):
    # for both images only th ek largest dct coefficients are kept
    dct_img = blockwise_dct(img, block_size)
    k_mask = k_largest_values_blockwise(dct_img, block_size, k)
    orig_img_dct = blockwise_dct(orig_img, block_size)
    k_mask_original = k_largest_values_blockwise(orig_img_dct, block_size, k)
    # estimate the watermark
    omega_estimate = estimate_watermark(k_mask, k_mask_original, block_size, k, alpha)
    print(omega_estimate)
    omega_mean = omega.mean()
    print(omega_mean)
    mean_omega_estimate = omega_estimate.mean()
    print(mean_omega_estimate)

    # the image does not contain a watermark since both the mystery and the original image are exactly the same
    if mean_omega_estimate == 0:
        return False

    gamma = get_gamma(omega_estimate, mean_omega_estimate, omega, omega_mean, k)
    print(gamma)

    # decide if image has a watermark based on its gamma value and the given threshold
    if gamma < threshold:
        return False
    else:
        return True


############################################# check if needs to be divided ###########################################
# returns the sum of the estimates watermark values
def estimate_watermark(k_mask, orig_image, block_size, k, alpha):
    sum_omega = np.array([0] * k)
    img_size = k_mask.shape
    count = 0
    # iterate through each block
    for i in np.r_[:img_size[0]: block_size]:
        for j in np.r_[:img_size[1]: block_size]:
            block = k_mask[i:i + block_size, j:j + block_size]  # block of mystery image
            orig_block = orig_image[i:i + block_size, j:j + block_size]  # block of original image
            # adding the estimated omega values
            sum_omega = np.add(sum_omega, estimate_omega_one_block(block, orig_block, k, alpha))
            count = count + 1

    return sum_omega/count  # return the average for each of the omega values


###################################################### cehck ###################################################
def estimate_omega_one_block(block, img_block, k, alpha):
    omega_est = []
    #myst_ph = block[0][0]  # keep for later (DC coefficient)
    #myst_ph[0][0] = 0  # will not be one of the k largest magnitude values
    block[0][0] = 0
    img_block[0][0] = 0

    # sort both the mystery image block and the original block by magnitude
    one_dim_myst = np.reshape(block, (-1))
    k_mystery = (abs(one_dim_myst).argsort())[::-1]

    #ph = img_block[0][0]  # keep for later (DC coefficient)
    #ph[0][0] = 0  # will not be one of the k largest magnitude values

    one_dim = np.reshape(img_block, (-1))
    k_original = (abs(one_dim).argsort())[::-1]

    for i in range(k):  # estimates wi for each of the k coefficients
        ci = one_dim[k_original[i]]  # coefficient of the image
        ci_hat = one_dim_myst[k_mystery[i]]  # coefficient of the mystery image
        wi = (ci_hat - ci) / (alpha * ci)  # wi_hat = (ci_hat - ci) / alpha * ci
        omega_est.append(wi)

    return omega_est


# computes the gamma value assuming that we know the original omega value
def get_gamma(omega_est, o_mean_est, orig_omega, orig_o_mean, k):
    num = 0
    den_est = 0
    den_omega = 0
    for i in range(k):
        num += ((omega_est[i] - o_mean_est) * (orig_omega[i] - orig_o_mean))
        den_est = (omega_est[i] - o_mean_est)**2
        den_omega = (orig_omega[i] - orig_o_mean) ** 2

    return num / (math.sqrt(den_est * den_omega))

