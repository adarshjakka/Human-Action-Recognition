import os
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load or generate data
if not os.path.exists("model/data.txt.npy"):
    from data_preparation import read_and_save_data
    read_and_save_data()

data = np.load("model/data.txt.npy", allow_pickle=True)
labels = np.load("model/labels.txt.npy")

X = [sample[1] for sample in data if len(sample) > 1]
X = np.array(X).reshape(-1, 5, 4, 3)
labels = to_categorical(labels[:len(X)])

# Shuffle and split
indices = np.arange(len(X))
np.random.shuffle(indices)
X, labels = X[indices], labels[indices]
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2)

# Load or train model
if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        classifier = model_from_json(json_file.read())
    classifier.load_weights("model/model_weights.h5")
else:
    classifier = Sequential([
        Conv2D(32, (1, 1), activation='relu', input_shape=(5, 4, 3)),
        MaxPooling2D(pool_size=(1, 1)),
        Conv2D(32, (1, 1), activation='relu'),
        MaxPooling2D(pool_size=(1, 1)),
        Flatten(),
        Dense(units=256, activation='relu'),
        Dense(units=labels.shape[1], activation='softmax')
    ])
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(X, labels, epochs=20, batch_size=16, verbose=2)
    with open('model/history.pckl', 'wb') as f:
        pickle.dump(history.history, f)
    with open('model/model.json', 'w') as f:
        f.write(classifier.to_json())
    classifier.save_weights('model/model_weights.h5')

# Evaluation
y_pred = np.argmax(classifier.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))
print("F1 Score:", f1_score(y_true, y_pred, average='macro'))
print("Accuracy:", accuracy_score(y_true, y_pred))