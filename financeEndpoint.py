from flask import Flask, jsonify, request
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load model and scaler
model_path = 'best_model.h5'
scaler_path = 'scaler.pkl'
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

# Label mapping
INDEX2LABEL = {0: 'Kurang Baik', 1: 'Baik', 2: 'Sangat Baik'}

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        try:
            # Get input from JSON
            data = request.get_json()
            penghasilan_bulanan = float(data['penghasilan_bulanan'])/1000000
            pengeluaran_bulanan = float(data['pengeluaran_bulanan'])/1000000
            tabungan_bulanan = float(data['tabungan_bulanan'])/1000000

            # Perform prediction
            new_input = [penghasilan_bulanan, pengeluaran_bulanan, tabungan_bulanan]
            predicted_label = predict_with_new_input(new_input)

            return jsonify({'predicted_label': predicted_label, 'dd':pengeluaran_bulanan})
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({'error': str(e)}), 400
    elif request.form:
        try:
            # Get input from form-data
            penghasilan_bulanan = float(request.form['penghasilan_bulanan'])/1000000
            pengeluaran_bulanan = float(request.form['pengeluaran_bulanan'])/1000000
            tabungan_bulanan = float(request.form['tabungan_bulanan'])/1000000

            # Perform prediction
            new_input = [penghasilan_bulanan, pengeluaran_bulanan, tabungan_bulanan]
            predicted_label = predict_with_new_input(new_input)

            return jsonify({'predicted_label': predicted_label, 'dd':pengeluaran_bulanan})
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({'error': str(e)}), 400
        

def predict_with_new_input(new_input):
    # Standardize new input
    new_input_scaled = scaler.transform([new_input])

    # Predict
    prediction = np.argmax(model.predict(new_input_scaled), axis=-1)

    # Convert prediction to original label
    predicted_label = INDEX2LABEL[prediction[0]]
    return predicted_label

if __name__ == '__main__':
    app.run(debug=True)
