# author:   Lena Luisa Feiler
# ID:       i6246119

import cv2
import numpy as np


def add_motion_blur(image_rgb, a, b):
    # extract all components for each image
    (r_img, g_img, b_img) = cv2.split(image_rgb)

    # calculate diagonal blur for each colour channel
    r_img_degraded = degrade_single_channel(r_img, a, b).astype(np.uint8)
    g_img_degraded = degrade_single_channel(g_img, a, b).astype(np.uint8)
    b_img_degraded = degrade_single_channel(b_img, a, b).astype(np.uint8)

    # merge the degraded colour channel to retrieve the resulting blurred image
    motion_blur_img = cv2.merge((b_img_degraded, g_img_degraded, r_img_degraded))

    return motion_blur_img


def degrade_single_channel(channel, a, b):
    channel = channel.astype(np.double)
    fourier_img = np.fft.fft2(channel)  # get channel into fourier domain

    h_blur = get_degrading_fun(channel, a, b)  # retrieve degrading function

    # apply degrading function to the channel to generate the motion blurry channel
    motion_blur = np.multiply(fourier_img, h_blur)
    motion_blur_img = np.fft.ifft2(motion_blur)  # transform channel back into time domain

    # normalize blurry channel before returning it
    return cv2.normalize(np.abs(motion_blur_img), None, 0, 255, cv2.NORM_MINMAX)


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


def add_gaussian_noise(img, mean, std):
    # split image in its three colour components
    (b_img, g_img, r_img) = cv2.split(img)

    ################
    # gauss_noise = random_noise(np.abs(img).astype(np.uint8), 'gaussian', mean, std)

    # calculate diagonal blur for each colour channel
    r_gauss = add_gaussian_single_channel(r_img, mean, std).astype(np.uint8)
    g_gauss = add_gaussian_single_channel(g_img, mean, std).astype(np.uint8)
    b_gauss = add_gaussian_single_channel(b_img, mean, std).astype(np.uint8)

    # merge the degraded colour channel to retrieve the resulting blurred image
    return cv2.merge((b_gauss, g_gauss, r_gauss))


def add_gaussian_single_channel(channel, mean, std):
    f_img = np.fft.fft2(channel)  # transfer channel into ff domain
    noise = np.random.normal(mean, std, channel.shape).astype(np.uint8)  # Gaussian Noise

    f_noise = np.fft.fft2(noise)  # transfer noise into ff domain
    noisy_img = np.add(f_img, f_noise)  # adding noise to the colour component

    noisy_img = np.fft.ifft2(noisy_img)  # transfer channel back into time domain

    # normalize the picture before returning it
    return cv2.normalize(np.abs(noisy_img), None, 0, 255, cv2.NORM_MINMAX)


def filter_all_channels_mmse(img, blur_img, a, b):
    # split both images in their three colour components
    (r_img, g_img, b_img) = cv2.split(img)
    (b_blur, g_blur, r_blur) = cv2.split(blur_img)

    # filter each component separately
    r_img_filtered = mmse_filter(r_img, r_blur, a, b)
    g_img_filtered = mmse_filter(g_img, g_blur, a, b)
    b_img_filtered = mmse_filter(b_img, b_blur, a, b)

    # merge the filtered components to retrieve the filtered image
    return cv2.merge((b_img_filtered, g_img_filtered, r_img_filtered))


def direct_inv_filter(img, a, b):
    # split image in its three colour components
    (r_img, g_img, b_img) = cv2.split(img)

    # F_hat = G/H
    # apply direct filtering to each component individually
    r_img_filtered = d_filter_single_channel(r_img, a, b).astype(np.uint8)
    g_img_filtered = d_filter_single_channel(g_img, a, b).astype(np.uint8)
    b_img_filtered = d_filter_single_channel(b_img, a, b).astype(np.uint8)

    # merge image before returning
    return cv2.merge((b_img_filtered, g_img_filtered, r_img_filtered))


def d_filter_single_channel(channel, a, b):
    channel = channel.astype(np.double)
    fourier_img = np.fft.fft2(channel)  # get channel into fourier domain

    h_blur = get_degrading_fun(channel, a, b)  # retrieve degrading function

    # apply degrading function to the channel to generate the motion blurry channel
    filtered_img = np.divide(fourier_img, h_blur)  # F_hat = G/H
    filtered_img = np.fft.ifft2(filtered_img)  # transform channel back into time domain

    # normalize blurry channel before returning it
    return cv2.normalize(np.abs(filtered_img), None, 0, 255, cv2.NORM_MINMAX)


####### clean up
def mmse_filter(img, blur_img, a, b):
    # split image in its three colour components
    (r_img, g_img, b_img) = cv2.split(img)
    (r_blur, g_blur, b_blur) = cv2.split(blur_img)

    # apply mmse filtering to each component individually
    r_img_filtered = mmse_single_channel(r_img, r_blur, a, b).astype(np.uint8)
    g_img_filtered = mmse_single_channel(g_img, g_blur, a, b).astype(np.uint8)
    b_img_filtered = mmse_single_channel(b_img, b_blur, a, b).astype(np.uint8)

    # merge image before returning
    return cv2.merge((b_img_filtered, g_img_filtered, r_img_filtered))


### clean up
def mmse_single_channel(channel, blur_channel, a, b):
    channel = channel.astype(np.double)
    noise = blur_channel - channel  # approximation of the noise

    ps_noise = abs(np.fft.fft2(noise)) ** 2  # power spectrum noise
    ps_img = abs(np.fft.fft2(channel)) ** 2  # power spectrum original image

    h = get_degrading_fun(channel, a, b)
    # h = np.random.normal(0.2, 1, channel.shape).astype('uint8')  # Gaussian Noise
    # Wiener filter
    h_w = np.conj(h) / (np.abs(h) ** 2 + ps_noise / ps_img)

    # apply Wiener Filter to image
    g = np.fft.fft2(blur_channel)
    filtered_img = h_w * g
    # transform to frequency domain
    filtered_img = np.abs(np.fft.ifft2(filtered_img))

    return cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)


# variables
alpha = -0.05
beta = 0.12

bgr_img = cv2.imread("iivp/pictures/exercise1/bird.jpg")  # read image
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)  # convert to rgb colour space

##################### exercise 1.1 ########################
# adding diagonal motion blurring degradation function
blurred_img = add_motion_blur(rgb_img, alpha, beta)  # both a, b != 0
cv2.imwrite('iivp/resultPictures/exercise1/M_BlurryImage.jpg', blurred_img)

##################### exercise 1.2 ########################

blur_gauss_img = add_gaussian_noise(blurred_img, 0.2, 1)
cv2.imwrite('iivp/resultPictures/exercise1/MG_BlurryImage.jpg', blur_gauss_img)

##################### exercise 2.1 ########################
filtered_motion_img = direct_inv_filter(blurred_img, alpha, beta)
cv2.imwrite('iivp/resultPictures/exercise1/DF_M_blur.jpg', filtered_motion_img)

##################### exercise 2.2 ########################
filtered_blurry_img = direct_inv_filter(blur_gauss_img, alpha, beta)
cv2.imwrite('iivp/resultPictures/exercise1/DF_MG_blur.jpg', filtered_blurry_img)

##################### exercise 2.3 ########################
only_gauss_img = add_gaussian_noise(rgb_img, 0.2, 1)
mmse_gauss_img = mmse_filter(rgb_img, only_gauss_img, alpha, beta)
cv2.imwrite('iivp/resultPictures/exercise1/MMSE_Gauss.jpg', mmse_gauss_img)

##################### exercise 2.4 ########################
mmse_mg_img = mmse_filter(rgb_img, blur_gauss_img, alpha, beta)
cv2.imwrite('iivp/resultPictures/exercise1/MMSE_MG.jpg', mmse_mg_img)
