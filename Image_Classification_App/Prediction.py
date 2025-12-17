import cv2
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

categories = ['cats', 'dogs']


# ---------------- Image Preprocessing ----------------
def preprocess_image(image_path):
    pet_img = cv2.imread(image_path, 0)  # grayscale
    pet_img = cv2.resize(pet_img, (50, 50))
    image = pet_img.flatten().astype(np.float32) / 255.0
    return image


# ---------------- Load Test Dataset (for metrics) ----------------
def load_test_data(test_folder="Dataset/test"):
    X_test = []
    y_test = []

    for label, category in enumerate(categories):
        folder = os.path.join(test_folder, category)

        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            try:
                image = preprocess_image(img_path)
                X_test.append(image)
                y_test.append(label)
            except:
                pass

    return np.array(X_test), np.array(y_test)


# ---------------- Prediction + Metrics ----------------
def predict_image_all_models(image_path):
    image = preprocess_image(image_path).reshape(1, -1)

    models = {
        "SVM": "models/svm_model.sav",
        "KNN": "models/knn_model.sav",
        "Tree": "models/dt_model.sav"
    }

    # Load test data ONCE
    X_test, y_test = load_test_data()

    results = {}

    for name, path in models.items():
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # ---- Single image prediction ----
        pred = model.predict(image)[0]
        pred_label = categories[pred]

        # ---- Dataset evaluation ----
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')

        results[name] = {
            "prediction": pred_label,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

    return results


# ---------------- CLI Test ----------------
if __name__ == "__main__":
    image_path = "test_image.jpg"

    results = predict_image_all_models(image_path)

    print(f"\nPredictions for: {image_path}")
    print("-" * 60)

    for model, data in results.items():
        print(
            f"{model:<10} | "
            f"Pred: {data['prediction']:<4} | "
            f"Acc: {data['accuracy']:.2f} | "
            f"Prec: {data['precision']:.2f} | "
            f"Rec: {data['recall']:.2f} | "
            f"F1: {data['f1']:.2f}"
        )
