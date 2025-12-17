import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from data_loader import load_images_from_folder

train_dir = 'Dataset/train'

# Load data (already normalized)
X, y = load_images_from_folder(train_dir)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, refit=True)
grid.fit(X, y)

model = grid.best_estimator_
print("Best SVM Params:", grid.best_params_)

with open('models/svm_model.sav', 'wb') as f:
    pickle.dump(model, f)
