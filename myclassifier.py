import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

train_dir = 'Dataset/train'
test_dir = 'Dataset/test'

categories = ['cats','dogs']


#------------------------First Step-----------------------------

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
test_data = load_images_from_folder(test_dir)

random.shuffle(train_data)
random.shuffle(test_data)
     
#------------------------Secound Step: Split-----------------------------
            
# Split features and labels for training
xtrain = np.array([item[0] for item in train_data]) / 255.0  # Normalize
ytrain = np.array([item[1] for item in train_data])

# Split features and labels for testing
xtest = np.array([item[0] for item in test_data]) / 255.0  # Normalize
ytest = np.array([item[1] for item in test_data]) 



# ------------------------Third Step: Train Model-----------------------------
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


#------------------------Fourth Step-----------------------------
pick = open('model.sav','rb')
model = pickle.load(pick)
pick.close()

predictions = model.predict(xtest)
accuracy = model.score(xtest, ytest)
print("Accuracy:", accuracy)

# Show confusion matrix and classification report
conf_matrix = confusion_matrix(ytest, predictions)
class_report = classification_report(ytest, predictions, target_names=categories)

print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

#------------------------Final Step: predict the image--------------------------------
def predict_image(image_path, model, categories):
    try:
        pet_img = cv2.imread(image_path, 0)  # Load image in grayscale
        pet_img = cv2.resize(pet_img, (50, 50)) 
        image = np.array(pet_img).flatten() / 255.0  
        prediction = model.predict([image])[0]  
        print(f"Prediction for '{image_path}': {categories[prediction]}")
        
        # Show the image
        plt.imshow(pet_img, cmap='gray')
        plt.title(f"Prediction: {categories[prediction]}")
        plt.show()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


# To loop through the xtest if needed else comment it
for i in range(min(10, len(xtest))): 
    try:
        mypet = np.array(xtest[i]).reshape(50, 50)
        plt.imshow(mypet, cmap='gray')
        plt.title(f"Prediction: {categories[predictions[i]]}, True Label: {categories[ytest[i]]}")
        plt.show()
        # plt.pause(3)  
        # plt.clf() 
    except Exception as e:
        pass
