from django.shortcuts import render
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "food_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load the trained MinMaxScaler
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
# Debugging the scaler
print("✅ Loading scaler...")
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

if os.path.exists(scaler_path):
    print("✅ Scaler file found!")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        print(f"✅ Loaded scaler: {scaler}")
else:
    print("❌ Scaler file missing!")

# Define column names (MUST match the dataset used in training)
column_names = ["Ordered", "Month", "Week", "Footfall", "Number of Plates Consumed"]

def predict(request):
    if request.method == "POST":
        try:
            # Get input values from the form
            ordered = request.POST.get("ordered")
            month = request.POST.get("month")
            week = request.POST.get("week")
            footfall = request.POST.get("footfall")
            plates_consumed = request.POST.get("plates_consumed")

            # Check if any input is missing
            if None in [ordered, month, week, footfall, plates_consumed] or "" in [ordered, month, week, footfall, plates_consumed]:
                return render(request, "prediction/result.html", {"error": "⚠️ Please fill all fields!"})

            # Convert inputs to float
            ordered = int(ordered)
            month = int(month)
            week = int(week)
            footfall = int(footfall)
            plates_consumed = int(plates_consumed)

            # Convert input to NumPy array (2D)
            X_new = np.array([[ordered, month, week, footfall, plates_consumed]])

            print(f"🔹 Raw Input: {X_new}")

            # Convert to DataFrame
            X_new_df = pd.DataFrame(X_new, columns=column_names)

            print(f"🔹 Input DataFrame (before scaling):\n{X_new_df}")

            # Check if scaler is working
            print(f"🔹 Scaler Min: {scaler.data_min_}, Max: {scaler.data_max_}")

            # Scale input data
            X_new_scaled = scaler.transform(X_new_df)


            print(f"🔹 Scaled Input:\n{X_new_scaled}")

            # Ensure input is correct shape for model
            if len(X_new_scaled.shape) == 1:
                X_new_scaled = X_new_scaled.reshape(1, -1)

            # Predict using trained model
            y_pred_new = model.predict(X_new_scaled)

            print(f"🔹 Raw Prediction Output: {y_pred_new}")

            # Ensure prediction is a scalar
            predicted_waste = round(y_pred_new[0], 2) if y_pred_new.ndim == 1 else round(y_pred_new[0][0], 2)

            print("Scaler Min:", scaler.data_min_)
            print("Scaler Max:", scaler.data_max_)

            print("Feature Importance:", model.feature_importances_)



            return render(request, "prediction/result.html", {"prediction": predicted_waste})

        except Exception as e:
            return render(request, "prediction/result.html", {"error": f"❌ Error: {str(e)}"})

    return render(request, "prediction/form.html")
