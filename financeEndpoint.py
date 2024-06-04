from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Muat model dan scaler
model_path = 'best_model.h5'
scaler_path = 'scaler.pkl'
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# Label mapping
INDEX2LABEL = {0: 'kurang', 1: 'baik', 2: 'sangat baik'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Ambil input dari form
        penghasilan_bulanan = float(request.form['penghasilan_bulanan'])
        pengeluaran_bulanan = float(request.form['pengeluaran_bulanan'])
        tabungan_bulanan = float(request.form['tabungan_bulanan'])

        # Lakukan prediksi
        new_input = [penghasilan_bulanan, pengeluaran_bulanan, tabungan_bulanan]
        predicted_label = predict_with_new_input(new_input)

        return render_template('result.html', predicted_label=predicted_label)

def predict_with_new_input(new_input):
    # Standarisasi input baru
    new_input_scaled = scaler.transform([new_input])

    # Prediksi
    prediction = np.argmax(model.predict(new_input_scaled), axis=-1)

    # Konversi prediksi menjadi label asli
    predicted_label = INDEX2LABEL[prediction[0]]
    return predicted_label

if __name__ == '__main__':
    app.run(debug=True)
