from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import pickle

app = Flask(__name__)

with open('label_encoder.pickle', 'rb') as le_file:
    le = pickle.load(le_file)
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)

selected_columns = ["Air Pollution", "Alcohol use", "Dust Allergy", "OccuPational Hazards",
                    "Genetic Risk", "chronic Lung Disease", "Balanced Diet", "Obesity",
                    "Smoking", "Passive Smoker", "Chest Pain", "Coughing of Blood",
                    "Fatigue", "Weight Loss", "Shortness of Breath", "Wheezing",
                    "Swallowing Difficulty", "Clubbing of Finger Nails", "Frequent Cold",
                    "Dry Cough", "Snoring"]

@app.route('/')
def home():
    return render_template('index1.html', columns=selected_columns)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_values = [float(x) for x in request.form.values()]
        if not all(1 <= val <= 9 for val in input_values):
            return render_template('index1.html', columns=selected_columns, error="Input values should be between 1-9.")
        input_features = pd.DataFrame([input_values], columns=selected_columns)
        input_features_scaled = scaler.transform(input_features)       
        prediction = model.predict(input_features_scaled)       
        predicted_label = le.inverse_transform(prediction)
        return render_template('result1.html', prediction=predicted_label[0])

if __name__ == '__main__':
    app.run(debug=True)