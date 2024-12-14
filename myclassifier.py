import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

dir = 'pet images'

categories = ['cats','dogs']

data = []

#------------------------First Step-----------------------------

for category in categories:
    path = os.path.join(dir,category)
    label=categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        pet_img=cv2.imread(imgpath,0)
        try:
            pet_img=cv2.resize(pet_img,(50,50))
            image = np.array(pet_img).flatten()
            data.append([image,label])
        except Exception as e:
            pass
        

pick_in=open('data1.pickle','wb')
pickle.dump(data,pick_in)
pick_in.close()
     
     
#------------------------Secound Step-----------------------------
             
pick_in=open('data1.pickle','rb')
data = pickle.load(pick_in)
pick_in.close()
   
random.shuffle(data)
features = []
labels = []

for feature,label in data:
    features.append(feature)
    labels.append(label)

# Normalize features
features = np.array(features) / 255.0  # Scale pixel values to [0, 1]

# # Split Data
xtrain,xtest,ytrain,ytest = train_test_split(features,labels,test_size = 0.2, random_state=42)


# ------------------------Third Step: Train Model with Hyperparameter Tuning-----------------------------
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(xtrain, ytrain)

model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)


pick = open('model.sav','wb')
pickle.dump(model,pick)
pick.close()


#------------------------Fourth Step-----------------------------

pick = open('model.sav','rb')
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

# Evaluation metrics
conf_matrix = confusion_matrix(ytest, prediction)
class_report = classification_report(ytest, prediction, target_names=categories)

print("Accuracy: ",accuracy)


for i in range(10): 
    mypet = np.array(xtest[i]).reshape(50, 50)
    plt.imshow(mypet, cmap='gray')
    plt.title(f"Prediction: {categories[prediction[i]]}, True Label: {categories[ytest[i]]}")
    plt.show()
    # plt.pause(3)  
    # plt.clf() 
