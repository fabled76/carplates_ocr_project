import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from src.evaluation import evaluate_model
from src.utils import load_dataset

# === Load and Split ===
dataset_folder = 'data_chars'
X, y, class_names = load_dataset(dataset_folder, resize=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train 1-NN Classifier ===
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# === Predict & Evaluate ===
y_pred = knn.predict(X_test)

joblib.dump(knn, 'models/ocr_knn_model.pkl')

evaluate_model(y_test, y_pred, class_names)
