from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Trained Logistic Regression Model
model = pickle.load(open("BCModel2.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index2.html")


@app.route('/predict', methods=['POST'])
def predict():

    # Read user inputs
    Hours_Studied = float(request.form['Hours_Studied'])
    Previous_Score = float(request.form['Previous_Score'])

    # Prediction
    prediction = model.predict(np.array([[Hours_Studied, Previous_Score]]))

    # Convert numeric prediction to text
    if prediction[0] == 1:
        result = "Student is Pass"
    else:
        result = "Student is Fail"

    return render_template(
        "index2.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)