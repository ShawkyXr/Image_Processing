import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import cv2
import numpy as np

from cnn_model import CNNModel
from rnn_model import RNNModel

categories = ['cats', 'dogs']

def load_data(folder_path):
    X = []
    y = []
    for label, category in enumerate(categories):
        folder = os.path.join(folder_path, category)
        if not os.path.isdir(folder):
            continue
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                # Read grayscale and resize to 50x50
                pet_img = cv2.imread(img_path, 0)
                if pet_img is not None:
                    pet_img = cv2.resize(pet_img, (50, 50))
                    # Normalize to [0, 1]
                    pet_img = pet_img.astype(np.float32) / 255.0
                    X.append(pet_img)
                    y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                pass
    return np.array(X), np.array(y)

def train_model(model, X_train, y_train, epochs=10, is_cnn=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare data based on model type
    if is_cnn:
        # CNN requires [batch, channels, H, W] -> [batch, 1, 50, 50]
        X_tensor = torch.tensor(X_train).unsqueeze(1)
    else:
        # RNN requires [batch, sequence, features] -> [batch, 50, 50]
        X_tensor = torch.tensor(X_train)
        
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    print("Loading data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "Dataset", "train")
    X_train, y_train = load_data(train_dir)
    print(f"Loaded {len(X_train)} samples.")

    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("\n--- Training CNN ---")
    cnn_model = CNNModel()
    train_model(cnn_model, X_train, y_train, epochs=5, is_cnn=True)
    torch.save(cnn_model.state_dict(), os.path.join(models_dir, "cnn_model.pth"))
    print("Saved CNN model.")

    print("\n--- Training RNN ---")
    rnn_model = RNNModel()
    train_model(rnn_model, X_train, y_train, epochs=5, is_cnn=False)
    torch.save(rnn_model.state_dict(), os.path.join(models_dir, "rnn_model.pth"))
    print("Saved RNN model.")
