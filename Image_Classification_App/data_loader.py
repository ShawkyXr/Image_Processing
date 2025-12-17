import os
import cv2
import numpy as np

categories = ['cats', 'dogs']

def load_images_from_folder(folder, img_size=(50, 50), normalize=True):
    X = []
    y = []

    for category in categories:
        path = os.path.join(folder, category)
        label = categories.index(category)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                pet_img = cv2.imread(img_path, 0)  # grayscale
                pet_img = cv2.resize(pet_img, img_size)

                image = pet_img.flatten().astype(np.float32)

                if normalize:
                    image /= 255.0   

                X.append(image)
                y.append(label)

            except Exception:
                pass

    return np.array(X), np.array(y)
