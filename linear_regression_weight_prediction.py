import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Create a dataset
data = {
    'Age': [3, 6, 9, 12, 15, 18],
    'Height': [80, 100, 120, 135, 160, 175],
    'Weight': [12, 22, 35, 50, 60, 70]
}

df = pd.DataFrame(data)

# Step 2: Define the features and target
X = df[['Age', 'Height']]  # Features: Age, Height
y = df['Weight']  # Target: Weight

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create the Linear Regression model
model = LinearRegression()

# Step 5: Train the model with the training data
model.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Predicted weights: {y_pred}")

# Step 8: Visualize the results
plt.scatter(df['Age'], df['Weight'], color='blue', label='Actual Weight')
plt.scatter(X_test['Age'], y_pred, color='red', label='Predicted Weight')
plt.xlabel('Age')
plt.ylabel('Weight')
plt.legend()
plt.show()
