from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    try:
        age = int(request.form['Age'])
        fever = int(request.form['Fever'])
        cough = int(request.form['Cough'])
        fatigue = int(request.form['Fatigue'])
        headache = int(request.form['Headache'])
        nausea = int(request.form['Nausea'])
        dizziness = int(request.form['Dizziness'])
        chest_pain = int(request.form['Chest_Pain'])
        breath = int(request.form['Shortness_of_Breath'])
        throat = int(request.form['Sore_Throat'])
        body_pain = int(request.form['Body_Pain'])
        bp = int(request.form['BP'])
        sugar = int(request.form['Sugar'])
        oxygen = int(request.form['Oxygen_Level'])

        input_data = np.array([[

            age,
            fever,
            cough,
            fatigue,
            headache,
            nausea,
            dizziness,
            chest_pain,
            breath,
            throat,
            body_pain,
            bp,
            sugar,
            oxygen

        ]])

        prediction = model.predict(input_data)

        disease = label_encoder.inverse_transform(prediction)

        probability = np.max(model.predict_proba(input_data)) * 100

        return render_template(
            "index.html",
            prediction_text=disease[0],
            probability=round(probability, 2)
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text="Error",
            probability=str(e)
        )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)