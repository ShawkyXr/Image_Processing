import cv2
import numpy as np

def denoise_image(img):
    
    denoised_img = cv2.medianBlur(img, 5)
    denoise_img_rgb = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB)

    return denoise_img_rgb