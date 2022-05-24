# author:   Lena Luisa Feiler
# ID:       i6246119

import cv2
import numpy as np

# variables
alpha = 0.05
beta = 0.12
mean = 0.2 #0.1  #0.05
std = 1 #0.02**0.5 #1


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


def degrade_single_channel(channel, a, b):
    channel = channel.astype(np.double)
    F = np.fft.fft2(channel)  # get channel into fourier domain

    H = get_degrading_fun(channel, a, b)  # retrieve degrading function

    # apply degrading function to the channel to generate the motion blurry channel
    G = np.multiply(F, H)
    g = np.fft.ifft2(G)  # transform channel back into time domain

    # normalize blurry channel before returning it
    return cv2.normalize(np.abs(g), None, 0, 255, cv2.NORM_MINMAX)


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


def add_gaussian_noise(img, mu, sigma):
    # split image in its three colour components
    (b_img, g_img, r_img) = cv2.split(img)

    # calculate diagonal blur for each colour channel
    r_gauss = add_gaussian_single_channel(r_img, mu, sigma).astype(np.uint8)
    g_gauss = add_gaussian_single_channel(g_img, mu, sigma).astype(np.uint8)
    b_gauss = add_gaussian_single_channel(b_img, mu, sigma).astype(np.uint8)

    # merge the degraded colour channel to retrieve the resulting blurred image
    return cv2.merge((b_gauss, g_gauss, r_gauss))


def add_gaussian_single_channel(channel, mu, sigma):
    f_img = np.fft.fft2(channel)  # transfer channel into ff domain
    noise = np.random.normal(mu, sigma, channel.shape).astype(np.uint8)  # Gaussian Noise
    f_noise = np.fft.fft2(noise)  # transfer noise into ff domain
    noisy_img = np.add(f_img, f_noise)  # adding noise to the colour component

    noisy_img = np.fft.ifft2(noisy_img)  # transfer channel back into time domain

    # normalize the picture before returning it
    return cv2.normalize(np.abs(noisy_img), None, 0, 255, cv2.NORM_MINMAX)


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


def dif_single_channel(channel, a, b):
    channel = channel.astype(np.double)
    fourier_img = np.fft.fft2(channel)  # get channel into fourier domain

    h_blur = get_degrading_fun(channel, a, b)  # retrieve degrading function

    # apply degrading function to the channel to generate the motion blurry channel
    filtered_img = np.divide(fourier_img, h_blur)  # F_hat = G/H
    filtered_img = np.fft.ifft2(filtered_img)  # transform channel back into time domain

    # normalize blurry channel before returning it
    return cv2.normalize(np.abs(filtered_img), None, 0, 255, cv2.NORM_MINMAX)


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


### clean up
def mmse_single_channel(channel, blur_channel, a, b, filter_motion_noise):
    channel = channel.astype(np.double)

    noise = channel - blur_channel  # approximation of the noise

    ps_noise = abs(np.fft.fft2(noise)) ** 2  # power spectrum noise
    ps_img = abs(np.fft.fft2(channel)) ** 2  # power spectrum original image

    # make sure that we do not divide by 0
    for i in range(ps_img.shape[0]):
        for j in range(ps_img.shape[1]):
            if ps_img[i][j] == 0:
                ps_img[i][j] = 0.001

    if filter_motion_noise:
        h = get_degrading_fun(channel, a, b)
    else:
        h = 1

    # approximate ratio between power spectrum of noise and original image by a constant
    k = sum(ps_noise) / sum(ps_img)
    #k = ps_noise / ps_img
    # print(k)
    # approximate k by the mean of the ratio
    # k = k.mean()
    # print(k)

    # Wiener filter
    h_w = np.conj(h) / (np.abs(h) ** 2 + k)

    # apply Wiener Filter to image
    g = np.fft.fft2(blur_channel)
    filtered_img = np.multiply(h_w, g)
    # transform to frequency domain
    filtered_img = np.abs(np.fft.ifft2(filtered_img))

    return cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX)


bgr_img = cv2.imread("iivp/pictures/bird.jpg")  # read image


##################### exercise 1.1 ########################
# adding diagonal motion blurring degradation function
blurred_img = add_motion_blur(bgr_img, alpha, beta)  # both a, b != 0
cv2.imwrite('iivp/resultPictures/exercise1/M_BlurryImage.jpg', blurred_img)


##################### exercise 1.2 ########################
blur_gauss_img = add_gaussian_noise(blurred_img, mean, std)
cv2.imwrite('iivp/resultPictures/exercise1/MG_BlurryImage.jpg', blur_gauss_img)

##################### exercise 2.1 ########################
filtered_motion_img = direct_inv_filter(blurred_img, alpha, beta)
cv2.imwrite('iivp/resultPictures/exercise1/DF_M_blur.jpg', filtered_motion_img)

##################### exercise 2.2 ########################
filtered_blurry_img = direct_inv_filter(blur_gauss_img, alpha, beta)
cv2.imwrite('iivp/resultPictures/exercise1/DF_MG_blur.jpg', filtered_blurry_img)


####################### exercise 2.3 ########################
only_gauss_img = add_gaussian_noise(bgr_img, mean, std)
cv2.imwrite('iivp/resultPictures/exercise1/G_BlurryImage.jpg', only_gauss_img)
mmse_gauss_img = mmse_filter(bgr_img, only_gauss_img, alpha, beta, False)
cv2.imwrite('iivp/resultPictures/exercise1/MMSE_Gauss.jpg', mmse_gauss_img)

##################### exercise 2.4 ########################
mmse_mg_img = mmse_filter(bgr_img, blur_gauss_img, alpha, beta, True)
cv2.imwrite('iivp/resultPictures/exercise1/MMSE_MG.jpg', mmse_mg_img)
