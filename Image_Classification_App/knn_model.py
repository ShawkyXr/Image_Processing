import pickle
from sklearn.neighbors import KNeighborsClassifier

from data_loader import load_images_from_folder

train_dir = 'Dataset/train'

X, y = load_images_from_folder(train_dir)

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'
)

knn.fit(X, y)

with open('models/knn_model.sav', 'wb') as f:
    pickle.dump(knn, f)
