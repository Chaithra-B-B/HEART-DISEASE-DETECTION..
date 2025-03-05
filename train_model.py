import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the dataset
# UCI Heart Disease Dataset URL
url = "D:\Downloads\heart.csv"

# Column names for the dataset
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Load the dataset
data = pd.read_csv(url, header=None, names=column_names, na_values="?")

# Step 2: Clean the dataset
# Drop rows with missing values
data = data.dropna()

# Step 3: Convert the target column (class) to binary (0 = No disease, 1 = Disease)
# Make sure that 'target' column is numeric
data['target'] = pd.to_numeric(data['target'], errors='coerce')

# Handle any remaining missing values after conversion (if any)
data = data.dropna(subset=['target'])

# Features (X) and Target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data['target']     # Target column (last column)

# Convert target to binary classification: 0 = No disease, 1 = Disease
y = (y > 0).astype(int)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save the trained model
joblib.dump(model, 'model_joblib_heart')
print("Model saved as 'model_joblib_heart'")
