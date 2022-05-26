# author:   Lena Luisa Feiler
# ID:       i6246119

import cv2
import numpy as np

# variables
alpha = -0.05
beta = 0.12
mean = 0.1
var = 1

bgr_img = cv2.imread("iivp/pictures/bird.jpg")  # read image
shape = bgr_img.shape
# noise is generated as values [0, 1], to make it visible we multiply with 255
gauss_noise = (np.random.normal(mean, var**0.5, (shape[0], shape[1])) * 255).astype(np.uint8)  # Gaussian Noise


# adds motion blur to an entire image
def add_motion_blur(image_bgr, a, b):
    # extract all components for each image
    (b_img, g_img, r_img) = cv2.split(image_bgr)

    # calculate diagonal blur for each colour channel
    r_img_degraded = degrade_single_channel(r_img, a, b).astype(np.uint8)
    g_img_degraded = degrade_single_channel(g_img, a, b).astype(np.uint8)
    b_img_degraded = degrade_single_channel(b_img, a, b).astype(np.uint8)

    # merge the degraded colour channel to retrieve the resulting blurred image
    motion_blur_img = cv2.merge((b_img_degraded, g_img_degraded, r_img_degraded))

    return motion_blur_img


# adds motion blur to a single colour channel
def degrade_single_channel(channel, a, b):
    channel = channel.astype(np.double)
    F = np.fft.fft2(channel)  # get channel into fourier domain

    H = get_degrading_fun(channel, a, b)  # retrieve degrading function

    # apply degrading function to the channel to generate the motion blurry channel
    G = np.multiply(F, H)
    g = np.fft.ifft2(G)  # transform channel back into time domain

    # normalize blurry channel before returning it
    return cv2.normalize(np.abs(g), None, 0, 255, cv2.NORM_MINMAX)


# retrieves the degrading function H
def get_degrading_fun(img, a, b):
    h, w = img.shape
    # get pixel in shape of the picture in range -1, 1
    [u, v] = np.mgrid[-h / 2:h / 2, -w / 2:w / 2]
    # adjust pixel to fit the image center
    u = 2 * u / h
    v = 2 * v / w

    # diagonal motion blurring degradation function
    h_blur = np.sinc((u * a + v * b)) * np.exp(-1j * np.pi * (u * a + v * b))

    return h_blur


# adds gaussian noise to an entire image
def add_gaussian_noise(img):
    # split image in its three colour components
    (b_img, g_img, r_img) = cv2.split(img)

    # calculate diagonal blur for each colour channel
    r_gauss = add_gaussian_single_channel(r_img).astype(np.uint8)
    g_gauss = add_gaussian_single_channel(g_img).astype(np.uint8)
    b_gauss = add_gaussian_single_channel(b_img).astype(np.uint8)

    # merge the degraded colour channel to retrieve the resulting blurred image
    return cv2.merge((b_gauss, g_gauss, r_gauss))


# adds gaussian noise to a single colour channel
def add_gaussian_single_channel(channel):
    f_img = np.fft.fft2(channel)  # transfer channel into ff domain
    f_noise = get_gauss_noise_fourier()  # get gaussian noise
    noisy_img = np.add(f_img, f_noise)  # adding noise to the colour component
    noisy_img = np.fft.ifft2(noisy_img)  # transfer channel back into time domain

    # normalize the picture before returning it
    return cv2.normalize(np.abs(noisy_img), None, 0, 255, cv2.NORM_MINMAX)


# retrieves the gaussian noise function N in the fourier domain
def get_gauss_noise_fourier():
    noise = gauss_noise.reshape(shape[0], shape[1])
    f_noise = np.fft.fft2(noise)  # transfer noise into ff domain
    return f_noise


# applies the DIF to an entire image
def direct_inv_filter(blurry_img, a, b):
    # split image in its three colour components
    (b_img, g_img, r_img) = cv2.split(blurry_img)

    # F_hat = G/H
    # apply direct filtering to each component individually
    r_img_filtered = dif_single_channel(r_img, a, b).astype(np.uint8)
    g_img_filtered = dif_single_channel(g_img, a, b).astype(np.uint8)
    b_img_filtered = dif_single_channel(b_img, a, b).astype(np.uint8)

    # merge image before returning
    return cv2.merge((b_img_filtered, g_img_filtered, r_img_filtered))


# applies the DIF to a single colour channel
def dif_single_channel(channel, a, b):
    channel = channel.astype(np.double)
    fourier_img = np.fft.fft2(channel)  # get channel into fourier domain

    h_blur = get_degrading_fun(channel, a, b)  # retrieve degrading function

    # apply degrading function to the channel to generate the motion blurry channel
    filtered_img = np.divide(fourier_img, h_blur)  # F_hat = G/H
    filtered_img = np.fft.ifft2(filtered_img)  # transform channel back into time domain

    # normalize blurry channel before returning it
    return cv2.normalize(np.abs(filtered_img), None, 0, 255, cv2.NORM_MINMAX)


# applies the MMSE channel to an entire image
def mmse_filter(img, blur_img, a, b, filter_motion_noise):
    # split image in its three colour components, image is in bgr format
    (b_img, g_img, r_img) = cv2.split(img)
    (b_blur, g_blur, r_blur) = cv2.split(blur_img)

    # apply MMSE filtering to each component individually
    r_img_filtered = mmse_single_channel(r_img, r_blur, a, b, filter_motion_noise).astype(np.uint8)
    g_img_filtered = mmse_single_channel(g_img, g_blur, a, b, filter_motion_noise).astype(np.uint8)
    b_img_filtered = mmse_single_channel(b_img, b_blur, a, b, filter_motion_noise).astype(np.uint8)

    # merge image before returning
    return cv2.merge((b_img_filtered, g_img_filtered, r_img_filtered))


# applies the MMSE filter to a single colour channel
def mmse_single_channel(channel, blur_channel, a, b, filter_motion_noise):
    channel = channel.astype(np.double)

    f_noise = get_gauss_noise_fourier()  # we assume we know noise, same as when generating the image

    ps_noise = abs(f_noise) ** 2  # power spectrum noise
    ps_img = abs(np.fft.fft2(channel)) ** 2  # power spectrum original image

    # make sure that we do not divide by 0
    ps_img = np.where(ps_img == 0, 0.000001, ps_img)

    if filter_motion_noise:  # exercise 4 (both motion blur and gaussian noise)
        # approximate ratio between power spectrum of noise and original image by a constant
        k = np.sum(ps_noise) / np.sum(ps_img)
        h = get_degrading_fun(channel, a, b)
        den = np.add(np.abs(h) ** 2, k)
    else:  # exercise 3: only additive noise
        h = 1  # h does not change F
        den = np.add(np.abs(h) ** 2, ps_noise / ps_img)  # no approximation is required

    # Wiener filter
    h_w = np.conj(h) / den

    # apply Wiener Filter to image
    g = np.fft.fft2(blur_channel)
    filtered_img = np.multiply(h_w, g)
    # transform to frequency domain
    filtered_img = np.abs(np.fft.ifft2(filtered_img))

    return cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)


##################### exercise 1.1 ########################
# adding diagonal motion blurring degradation function
blurred_img = add_motion_blur(bgr_img, alpha, beta)  # both a, b != 0
cv2.imwrite('iivp/resultPictures/exercise1/M_BlurryImage.jpg', blurred_img)


##################### exercise 1.2 ########################
blur_gauss_img = add_gaussian_noise(blurred_img)
cv2.imwrite('iivp/resultPictures/exercise1/MG_BlurryImage.jpg', blur_gauss_img)

##################### exercise 2.1 ########################
filtered_motion_img = direct_inv_filter(blurred_img, alpha, beta)
cv2.imwrite('iivp/resultPictures/exercise1/DF_M_blur.jpg', filtered_motion_img)

##################### exercise 2.2 ########################
filtered_blurry_img = direct_inv_filter(blur_gauss_img, alpha, beta)
cv2.imwrite('iivp/resultPictures/exercise1/DF_MG_blur.jpg', filtered_blurry_img)

####################### exercise 2.3 ########################
only_gauss_img = add_gaussian_noise(bgr_img)
cv2.imwrite('iivp/resultPictures/exercise1/G_BlurryImage.jpg', only_gauss_img)
mmse_gauss_img = mmse_filter(bgr_img, only_gauss_img, alpha, beta, False)
cv2.imwrite('iivp/resultPictures/exercise1/MMSE_Gauss.jpg', mmse_gauss_img)

##################### exercise 2.4 ########################
mmse_mg_img = mmse_filter(bgr_img, blur_gauss_img, alpha, beta, True)
cv2.imwrite('iivp/resultPictures/exercise1/MMSE_MG.jpg', mmse_mg_img)