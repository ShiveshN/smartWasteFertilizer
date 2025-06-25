import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('balanced_sensor_fertilizer_data.csv')

# Split data into features (X) and target (y)
X = df[["Sensor1", "Sensor2", "Sensor3", "Sensor4"]]
y = df["FertilizerConverted"].map({"Yes": 1, "No": 0})  # Encode labels as 1/0

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to classify user input
def classify_sensor_values():
    print("\nEnter sensor values (range 0-1023):")
    try:
        # Get input for four sensor values
        sensor_values = []
        for i in range(1, 5):
            value = int(input(f"Sensor{i}: "))
            if 0 <= value <= 1023:
                sensor_values.append(value)
            else:
                raise ValueError("Sensor value must be in the range 0-1023.")
        
        # Predict the classification
        prediction = rf.predict([sensor_values])[0]
        result = "Yes" if prediction == 1 else "No"
        print(f"\nFertilizer Converted: {result}")
    except ValueError as e:
        print(f"Invalid input: {e}")

# Allow the user to input sensor values for classification
classify_sensor_values()
