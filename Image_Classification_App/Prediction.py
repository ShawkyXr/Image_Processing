import cv2
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

categories = ['cats', 'dogs']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------- Image Preprocessing ----------------
def preprocess_image(image_path):
    pet_img = cv2.imread(image_path, 0)  # grayscale
    pet_img = cv2.resize(pet_img, (50, 50))
    image = pet_img.flatten().astype(np.float32) / 255.0
    return image


# ---------------- Load Test Dataset (for metrics) ----------------
def load_test_data(test_folder="Dataset/test"):
    if not os.path.isabs(test_folder):
        test_folder = os.path.join(SCRIPT_DIR, test_folder)

    X_test = []
    y_test = []

    for label, category in enumerate(categories):
        folder = os.path.join(test_folder, category)

        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Test folder not found: {folder}")

        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            try:
                image = preprocess_image(img_path)
                X_test.append(image)
                y_test.append(label)
            except Exception:
                pass

    return np.array(X_test), np.array(y_test)


# ---------------- Prediction + Metrics ----------------
def predict_image_all_models(image_path, selected_models=None):
    image_ml = preprocess_image(image_path).reshape(1, -1)
    
    # Preprocess for DL models if needed
    pet_img = cv2.imread(image_path, 0)
    pet_img = cv2.resize(pet_img, (50, 50))
    image_dl = pet_img.astype(np.float32) / 255.0

    models_ml = {
        "SVM": os.path.join(SCRIPT_DIR, "models/svm_model.sav"),
        "KNN": os.path.join(SCRIPT_DIR, "models/knn_model.sav"),
        "Tree": os.path.join(SCRIPT_DIR, "models/dt_model.sav")
    }

    # Load test data ONCE
    X_test_ml, y_test = load_test_data()

    results = {}

    for name, path in models_ml.items():
        if selected_models and name not in selected_models:
            continue
            
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # ---- Single image prediction ----
        pred = model.predict(image_ml)[0]
        pred_label = categories[pred]

        # ---- Dataset evaluation ----
        y_pred = model.predict(X_test_ml)

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
        
    try:
        import torch
        from cnn_model import CNNModel
        from rnn_model import RNNModel
        
        # Load DL test data
        X_test_dl = X_test_ml.reshape(-1, 50, 50) # assuming X_test_ml was flattened
        
        models_dl = {}
        if not selected_models or "CNN" in selected_models:
            cnn = CNNModel()
            cnn.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "models/cnn_model.pth"), weights_only=True))
            cnn.eval()
            models_dl["CNN"] = cnn
            
        if not selected_models or "RNN" in selected_models:
            rnn = RNNModel()
            rnn.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "models/rnn_model.pth"), weights_only=True))
            rnn.eval()
            models_dl["RNN"] = rnn
            
        for name, model in models_dl.items():
            with torch.no_grad():
                # Single prediction
                if name == "CNN":
                    x_single = torch.tensor(image_dl).unsqueeze(0).unsqueeze(0)
                else:
                    x_single = torch.tensor(image_dl).unsqueeze(0)
                    
                out = model(x_single)
                pred = torch.argmax(out, dim=1).item()
                pred_label = categories[pred]
                
                # Dataset evaluation
                if name == "CNN":
                    X_test_tensor = torch.tensor(X_test_dl).unsqueeze(1)
                else:
                    X_test_tensor = torch.tensor(X_test_dl)
                    
                out_test = model(X_test_tensor)
                y_pred = torch.argmax(out_test, dim=1).numpy()
                
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
                rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

                results[name] = {
                    "prediction": pred_label,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1
                }
    except Exception as e:
        print(f"DL evaluation error: {e}")

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
