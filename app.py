import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Data contoh: penghasilan, pengeluaran, tabungan, dan label kesehatan keuangan
income = np.array([5000, 6000, 4000, 7000, 8000])
expenses = np.array([3000, 2500, 3500, 2000, 2800])
savings = np.array([2000, 3500, 500, 5000, 5200])
# Label kesehatan keuangan: 0 - baik, 1 - sedang, 2 - kurang
financial_health = np.array([0, 0, 2, 1, 0])  

# Menggabungkan tiga array menjadi satu array dengan tiga kolom
X = np.column_stack((income, expenses, savings))

# Inisialisasi k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True)

# List untuk menyimpan akurasi dari setiap fold
accuracies = []

# Iterasi melalui setiap fold
for train_index, test_index in kfold.split(X):
    # Bagi data menjadi set pelatihan dan pengujian
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = financial_health[train_index], financial_health[test_index]

    # Membuat model Sequential
    model = Sequential([
        Dense(128, activation='relu', input_shape=(3,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])

    # Kompilasi model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Melatih model
    model.fit(x=X_train, y=y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[reduce_lr, early_stopping], verbose=1)

    # Evaluasi model pada data pengujian
    _, accuracy = model.evaluate(X_test, y_test)
    accuracies.append(accuracy)

# Menampilkan akurasi rata-rata dari k-fold cross-validation
print("Rata-rata akurasi:", np.mean(accuracies))

# Prediksi kesehatan keuangan
new_data = np.array([[5500, 3200, 2500]])  # Contoh data baru
prediction = model.predict(new_data)
predicted_class = np.argmax(prediction)  # Ambil indeks kelas dengan probabilitas tertinggi
if predicted_class == 0:
    print("Kesehatan keuangan: Baik")
elif predicted_class == 1:
    print("Kesehatan keuangan: Sedang")
else:
    print("Kesehatan keuangan: Kurang")
