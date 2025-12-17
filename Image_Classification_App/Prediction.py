import cv2
import numpy as np
import pickle

categories = ['cats', 'dogs']

def preprocess_image(image_path):
    pet_img = cv2.imread(image_path, 0)
    pet_img = cv2.resize(pet_img, (50, 50))
    image = np.array(pet_img).flatten() / 255.0
    return image

def predict_image_all_models(image_path):
    image = preprocess_image(image_path)

    models = {
        "SVM": "models/svm_model.sav",
        "KNN": "models/knn_model.sav",
        "Decision Tree": "models/dt_model.sav"
    }


    predictions = {}

    for name, path in models.items():
        pick = open(path, 'rb')      
        model = pickle.load(pick)
        pick.close()

        pred = model.predict([image])[0]
        predictions[name] = categories[pred]

    return predictions

if __name__ == "__main__":
    image_path = "test_image.jpg"  

    results = predict_image_all_models(image_path)

    print(f"\nPredictions for: {image_path}")
    print("-" * 40)

    for model, pred in results.items():
        print(f"{model:<15}: {pred}")
