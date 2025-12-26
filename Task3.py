import tensorflow as tf
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# =========================
# LOAD DATASET (ONLINE)
# =========================
(dataset_train, dataset_test), info = tf.keras.datasets.cats_vs_dogs.load_data()

print("Dataset loaded successfully")

# =========================
# PREPROCESS DATA
# =========================
IMG_SIZE = 64
MAX_SAMPLES = 2000  # limit for fast execution

X = []
y = []

for i, (img, label) in enumerate(dataset_train):
    if i >= MAX_SAMPLES:
        break

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    X.append(gray.flatten())
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN SVM
# =========================
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# DISPLAY SAMPLE RESULTS
# =========================
plt.figure(figsize=(8, 4))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
    label = "Dog" if y_pred[i] == 1 else "Cat"
    plt.title(label)
    plt.axis("off")

plt.tight_layout()
plt.show()
