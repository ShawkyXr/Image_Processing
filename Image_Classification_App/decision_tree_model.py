import pickle
from sklearn.tree import DecisionTreeClassifier

from data_loader import load_images_from_folder

train_dir = 'Image_Classification_App/Dataset/train'

X, y = load_images_from_folder(train_dir)

dt = DecisionTreeClassifier(
    max_depth=20,
    random_state=42
)

dt.fit(X, y)

with open('Image_Classification_App/models/dt_model.sav', 'wb') as f:
    pickle.dump(dt, f)
