import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Memuat file CSV ke dalam DataFrame pandas
file_path = 'dataset.csv'
df = pd.read_csv(file_path)

# Mengubah label 'kesehatan tabungan' menjadi angka
LABEL2INDEX = {'kurang': 0, 'baik': 1, 'sangat baik': 2}
INDEX2LABEL = {v: k for k, v in LABEL2INDEX.items()}
df['kesehatan tabungan'] = df['kesehatan tabungan'].apply(lambda lab: LABEL2INDEX[lab])

# Memilih fitur dan label
X = df[['Penghasilan Bulanan', 'Pengeluaran Bulanan', 'Tabungan Bulanan']]
y = df['kesehatan tabungan']

# Standarisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Mengatur K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Menyimpan hasil dari setiap fold
fold_accuracies = []
fold_f1_scores = []
fold_precision_scores = []
fold_recall_scores = []
histories = []

# Model initialization (outside the loop)
best_model = None
best_accuracy = 0

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Training fold {fold + 1}")
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Membangun model neural network
    model = Sequential()
    model.add(Dense(32, input_dim=3, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Melatih model dan menyimpan history
    history = model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    histories.append(history)
    
    # Prediksi pada set pengujian
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    
    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    fold_f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    fold_precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    fold_recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    
    # Print confusion matrix dan classification report untuk setiap fold
    print(f"Fold {len(fold_accuracies)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL2INDEX.keys()))
    print("\n")
    
    # Save the best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Rata-rata hasil dari semua fold
print(f'Average Accuracy: {np.mean(fold_accuracies)}')
print(f'Average F1 Score: {np.mean(fold_f1_scores)}')
print(f'Average Precision Score: {np.mean(fold_precision_scores)}')
print(f'Average Recall Score: {np.mean(fold_recall_scores)}')

# Simpan model terbaik
model_path = 'best_model.h5'
best_model.save(model_path)
print(f"Model disimpan di {model_path}")

# Simpan scaler
import joblib
scaler_path = 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler disimpan di {scaler_path}")

# Plotting history training untuk setiap fold
for i, history in enumerate(histories):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.title(f'Fold {i+1} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.title(f'Fold {i+1} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()

# Fungsi untuk prediksi dengan model yang dimuat ulang
def predict_with_new_input(model_path, new_input):
    # Muat model yang telah disimpan
    model = tf.keras.models.load_model(model_path)
    
    # Standarisasi input baru
    new_input_scaled = scaler.transform([new_input])
    
    # Prediksi
    prediction = np.argmax(model.predict(new_input_scaled), axis=-1)
    
    # Konversi prediksi menjadi label asli
    predicted_label = INDEX2LABEL[prediction[0]]
    return predicted_label

# Contoh penggunaan prediksi dengan input baru
new_input = [8.0, 6.5, 1.5]  # Contoh input baru
predicted_label = predict_with_new_input(model_path, new_input)
print(f"Predicted label for new input {new_input}: {predicted_label}")
