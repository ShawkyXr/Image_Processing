import cv2
import numpy as np
import math

def gaussian_filter(data, filter_size, sigma=1.0):

    kernel = np.zeros((filter_size, filter_size))
    center = filter_size // 2

    
    for i in range(filter_size):
        for j in range(filter_size):
            diff = (i - center)**2 + (j - center)**2
            kernel[i, j] = math.exp(-diff / (2 * sigma**2))

    
    kernel /= kernel.sum()

    # Apply the Gaussian filter to the image
    data_final = np.zeros_like(data, dtype=float)
    padded_data = np.pad(data, ((center, center), (center, center), (0, 0)), mode='constant')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]): 
                region = padded_data[i:i + filter_size, j:j + filter_size, k]
                data_final[i, j, k] = np.sum(region * kernel)

    return data_final.astype(np.uint8)

def denoise_image(img):

    filter_size = 5
    sigma = 1.0
    denoised_img = gaussian_filter(img, filter_size, sigma)

    denoise_img_rgb = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)

    return denoise_img_rgb
