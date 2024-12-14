import cv2
import numpy as np

def mean_filter(data, filter_size):
    
    pad_size = filter_size // 2

    
    padded_data = np.pad(data, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')

    
    data_final = np.zeros_like(data, dtype=float)

    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):  # For each color channel (if RGB image)
                # Extract the filter region
                region = padded_data[i:i + filter_size, j:j + filter_size, k]
                # Compute the mean and assign it to the center pixel
                data_final[i, j, k] = np.mean(region)

    return data_final.astype(np.uint8)

def denoise_image(img):
    # Apply the custom mean filter
    filter_size = 5  
    denoised_img = mean_filter(img, filter_size)

    # Convert the result to RGB format
    denoise_img_rgb = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)

    return denoise_img_rgb
