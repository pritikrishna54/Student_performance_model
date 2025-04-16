from flask import Flask, request, render_template
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("student_performance_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    hours_studied = float(request.form["hours_studied"])
    attendance_percentage = float(request.form["attendance_percentage"])
    sleep_hours = float(request.form["sleep_hours"])
    parental_education_level = request.form["parental_education_level"]
    extracurricular_participation = request.form["extracurricular_participation"]
    internet_access = request.form["internet_access"]

    # Encode categorical features
    edu_encoded = label_encoders['parental_education_level'].transform([parental_education_level])[0]
    extra_encoded = label_encoders['extracurricular_participation'].transform([extracurricular_participation])[0]
    net_encoded = label_encoders['internet_access'].transform([internet_access])[0]
     # Feature array
    features = np.array([[hours_studied, attendance_percentage, sleep_hours,
                          edu_encoded, extra_encoded, net_encoded]])

    # Predict
    prediction = model.predict(features)[0]
    performance_label = label_encoders['performance'].inverse_transform([prediction])[0]
    return render_template("index.html", prediction=performance_label)
if __name__ == "__main__":
    app.run(debug=True)
