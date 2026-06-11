# ----------------------------------------
# Predictive Analytics Using Historical Data
# ----------------------------------------

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ----------------------------------------
# Load Dataset
# ----------------------------------------

df = pd.read_csv("sales_data.csv")

print("\nDataset Preview:")
print(df.head())

# ----------------------------------------
# Data Cleaning
# ----------------------------------------

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# ----------------------------------------
# Feature Selection
# ----------------------------------------

X = df[["Month"]]
y = df["Sales"]

# ----------------------------------------
# Split Dataset
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ----------------------------------------
# Train Regression Model
# ----------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# ----------------------------------------
# Predictions
# ----------------------------------------

y_pred = model.predict(X_test)

# Future Prediction
future_months = pd.DataFrame({
    "Month": [13, 14, 15, 16]
})

future_predictions = model.predict(future_months)

# ----------------------------------------
# Model Evaluation
# ----------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2 Score : {r2:.2f}")

# ----------------------------------------
# Future Forecast Results
# ----------------------------------------

print("\nFuture Sales Predictions:")

for month, prediction in zip(
    future_months["Month"],
    future_predictions
):
    print(f"Month {month}: {prediction:.2f}")

# ----------------------------------------
# Visualization
# ----------------------------------------

plt.figure(figsize=(10, 6))

# Original Data
sns.scatterplot(
    x=df["Month"],
    y=df["Sales"],
    color="blue",
    s=100,
    label="Actual Sales"
)

# Regression Line
plt.plot(
    X,
    model.predict(X),
    color="red",
    linewidth=2,
    label="Regression Line"
)

# Future Predictions
sns.scatterplot(
    x=future_months["Month"],
    y=future_predictions,
    color="green",
    s=120,
    label="Predicted Sales"
)

plt.title("Sales Forecast Prediction")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()

plt.grid(True)

plt.show()