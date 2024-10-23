# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib  # for saving the model

# Load the CSV file
df = pd.read_csv('accident.csv')

# Preview the first few rows of the dataset
print(df.head())

# Ensure there are no missing values in the relevant columns
df = df.dropna(subset=['speed_limit', 'weather', 'road_type', 'vehicle_type', 'num_vehicles', 'time_of_day', 'severity'])

# Define dependent variable (y) and independent variables (X)
X = df[['speed_limit', 'weather', 'road_type', 'vehicle_type', 'num_vehicles', 'time_of_day']]
y = df['severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse:.2f}")

# Save the model
joblib.dump(model, 'accident_severity_model.pkl')
print("Model saved as 'accident_severity_model.pkl'")

# Hypothetical independent variables for a new accident case
new_accident = np.array([[50, 1, 2, 3, 2, 15]])  # speed_limit, weather, road_type, vehicle_type, num_vehicles, time_of_day

# Ensure the new input matches the trained data's feature structure
new_accident = new_accident.reshape(1, -1)

# Predict the accident severity for the new data
severity_prediction = model.predict(new_accident)
print(f"Predicted Accident Severity for the new case: {severity_prediction[0]:.2f}")

