import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
# Check for missing values
print(data.isnull().sum())

# Fill missing values or drop rows/columns if necessary
data = data.dropna()  # Dropping rows with missing values for simplicity

# Feature selection: Selecting relevant features for prediction
features = ['Genre', 'Director', 'Actor 1','Actor 2','Actor 3']
target = 'Rating'

X = data[features]
y = data[target]

# One-hot encoding for categorical variables
X_encoded = pd.get_dummies(X, columns=features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Feature scaling (if necessary)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training using Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions on the test set
y_pred = model.predict(X_test_scaled)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Example of predicting a new movie rating (replace with actual feature values)
new_movie_data = pd.DataFrame({
    'Genre_Action': [1],
    'Genre_Comedy': [0],
    'Genre_Drama': [0],
    # Add one-hot encoded features for Director and Actors as well...
})

# Ensure new_movie_data has the same columns as X_encoded after one-hot encoding
# Create a list of all columns used in training
all_columns = X_encoded.columns.tolist()

# Reindex new_movie_data to include all columns, filling missing ones with 0
new_movie_data = new_movie_data.reindex(columns=all_columns, fill_value=0)

# Scale the new movie data
new_movie_scaled = scaler.transform(new_movie_data)
predicted_rating = model.predict(new_movie_scaled)

print(f'Predicted Rating: {predicted_rating[0]:.2f}')

