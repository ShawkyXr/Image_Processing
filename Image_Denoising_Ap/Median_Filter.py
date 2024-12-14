import numpy as np
import cv2

def denoise_image(img):
    
    if len(img.shape) == 3 and img.shape[2] == 3: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    filter_size = 5
    denoised_img = median_filter(img, filter_size)
    
    denoised_img = denoised_img.astype(np.uint8)
    denoise_img_rgb = cv2.cvtColor(cv2.merge([denoised_img, denoised_img, denoised_img]), cv2.COLOR_BGR2RGB)
    
    print(type(denoise_img_rgb))
    return denoise_img_rgb


# Median filter function
def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final
