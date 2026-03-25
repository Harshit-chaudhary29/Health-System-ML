# # diabetes_app.py
# import pandas as pd
# import numpy as np
# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # ---------------------------
# # Load Dataset
# # ---------------------------
# @st.cache_data
# def load_data():
#     # Download dataset from Kaggle or UCI and keep as diabetes.csv
#     data = pd.read_csv("diabetes.csv")
#     return data

# data = load_data()

# st.title("🩺 Diabetes Prediction System")
# st.write("Predict whether a patient is diabetic using health parameters.")

# st.subheader("Dataset Preview")
# st.dataframe(data.head())

# # ---------------------------
# # Train Model
# # ---------------------------
# X = data.drop("Outcome", axis=1)  # features
# y = data["Outcome"]              # target

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Accuracy
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# st.write(f"✅ Model trained with accuracy: **{acc*100:.2f}%**")

# # ---------------------------
# # User Input
# # ---------------------------
# st.subheader("Enter Patient Details")

# pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
# glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
# blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
# skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
# insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
# bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
# dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
# age = st.number_input("Age", min_value=1, max_value=120, value=30)

# # ---------------------------
# # Prediction
# # ---------------------------
# if st.button("Predict"):
#     user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
#                            insulin, bmi, dpf, age]])
#     user_data = scaler.transform(user_data)
#     prediction = model.predict(user_data)

#     if prediction[0] == 1:
#         st.error("⚠️ The model predicts: **Diabetic**")
#     else:
#         st.success("✅ The model predicts: **Not Diabetic**")




# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# app = Flask(__name__)

# # Load dataset
# data = pd.read_csv("diabetes.csv")

# # Features & labels
# X = data.drop("Outcome", axis=1)
# y = data["Outcome"]

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# @app.route("/")
# def home():
#     return render_template("diabetes.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Collect form data
#         values = [float(request.form[key]) for key in request.form]
#         input_data = np.array(values).reshape(1, -1)

#         # Prediction
#         prediction = model.predict(input_data)[0]
#         result = "Diabetic" if prediction == 1 else "Not Diabetic"

#         return render_template("diabetes.html", prediction=result)
#     except Exception as e:
#         return render_template("diabetes.html", prediction=f"Error: {str(e)}")

# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# app = Flask(__name__)

# # Load dataset
# data = pd.read_csv("diabetes.csv")

# # Features & labels
# X = data.drop("Outcome", axis=1)
# y = data["Outcome"]

# # Train model
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)


# @app.route("/")
# def home():
#     return render_template("diabetes.html")


# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Extract values from the form
#         form_data = request.form.to_dict()

#         # Handle systolic and diastolic BP
#         systolic = float(form_data.get("Systolic", 0))
#         diastolic = float(form_data.get("Diastolic", 0))

#         # Convert into single BloodPressure value (average)
#         blood_pressure = (systolic + diastolic) / 2

#         # Build feature list in the correct order (as in dataset)
#         values = [
#             float(form_data.get("Pregnancies", 0)),
#             float(form_data.get("Glucose", 0)),
#             blood_pressure,
#             float(form_data.get("SkinThickness", 0)),
#             float(form_data.get("Insulin", 0)),
#             float(form_data.get("BMI", 0)),
#             float(form_data.get("DiabetesPedigreeFunction", 0)),
#             float(form_data.get("Age", 0)),
#         ]

#         # Convert to numpy array for prediction
#         input_data = np.array(values).reshape(1, -1)

#         # Prediction
#         prediction = model.predict(input_data)[0]
#         result = "🔴 Diabetic" if prediction == 1 else "🟢 Not Diabetic"

#         return render_template("diabetes.html", prediction=result)

#     except Exception as e:
#         return render_template("diabetes.html", prediction=f"⚠️ Error: {str(e)}")


# if __name__ == "__main__":
#     app.run(debug=True)





# diabetes_app.py
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# --- Load Model and Scaler ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and Scaler loaded successfully.")
except FileNotFoundError:
    print("ERROR: model.pkl or scaler.pkl not found. Run 'train_and_save_model.py' first.")
    # Exit or handle gracefully if critical files are missing
    # For a robust app, you might want to stop the server here.

# The feature order must match the training data:
# ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route("/")
def home():
    """Renders the main input form."""
    # Pass 'None' for the result on the initial load
    return render_template("diabetes.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    """Handles the form submission and returns the prediction."""
    result = {
        "prediction_text": "Error during prediction.",
        "result_class": "error"
    }
    
    try:
        # Extract and convert form data
        form_data = request.form.to_dict()
        
        # Calculate single BloodPressure from Systolic and Diastolic
        systolic = float(form_data.get("Systolic", 0))
        diastolic = float(form_data.get("Diastolic", 0))
        # Use a simple average or just the systolic/diastolic if that's what the model expects
        # Based on your previous snippet, the model expects a single 'BloodPressure' value.
        # We will use the average as before. If your model uses only one, adjust this line.
        blood_pressure = (systolic + diastolic) / 2.0
        if blood_pressure == 0:
             # Prevent division by zero if both are zero
             blood_pressure = 0
        
        # Build feature list in the EXACT order the model was trained on
        values = [
            float(form_data.get("Pregnancies", 0)),
            float(form_data.get("Glucose", 0)),
            blood_pressure, # Calculated BP
            float(form_data.get("SkinThickness", 0)),
            float(form_data.get("Insulin", 0)),
            float(form_data.get("BMI", 0)),
            float(form_data.get("DiabetesPedigreeFunction", 0)),
            float(form_data.get("Age", 0)),
        ]

        # Convert to numpy array and reshape for prediction (1 sample, 8 features)
        input_data = np.array(values).reshape(1, -1)

        # Scale the input data using the saved scaler
        scaled_input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input_data)[0]

        if prediction == 1:
            result["prediction_text"] = "⚠️ Patient is **Diabetic** (Positive Result)"
            result["result_class"] = "positive"
        else:
            result["prediction_text"] = "✅ Patient is **Non-Diabetic** (Negative Result)"
            result["result_class"] = "negative"

    except Exception as e:
        print(f"An error occurred: {e}")
        # The 'result' dictionary already contains the error message
        pass

    # Render the template again, passing the prediction result
    return render_template("diabetes.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)

