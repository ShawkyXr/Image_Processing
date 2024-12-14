import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

train_dir = '/home/hossam/work/Image_Processing/Dataset/train'
categories = ['cats','dogs']


def load_images_from_folder(folder):
    data = []
    path = []
    label =[]
    
    for category in categories:
        path = os.path.join(folder,category)
        label=categories.index(category)
        
        for img in os.listdir(path):
            imgpath = os.path.join(path,img)
            try:
                pet_img=cv2.imread(imgpath,0)
                pet_img=cv2.resize(pet_img,(50,50))
                image = np.array(pet_img).flatten()
                data.append([image,label])
            except Exception as e:
                pass
    return data 


train_data = load_images_from_folder(train_dir)
random.shuffle(train_data)

xtrain = np.array([item[0] for item in train_data]) / 255.0  # Normalize
ytrain = np.array([item[1] for item in train_data])

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0)
grid.fit(xtrain, ytrain)

model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)


pick = open('model.sav','wb')
pickle.dump(model,pick)
pick.close()
