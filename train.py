import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


# Load dataset
data = pd.read_csv('dataset/patients_dataset.csv')

# print(data.head())
# print(data.columns)
# print(data.shape)

# All columns in the dataset
['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']

target_col = 'diabetes'
features_col = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]


features = data[features_col]
target = data[target_col]


# Splitting the dataset into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Training the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(features_train, target_train)

# Evaluating the model
y_pred = model.predict(features_test)
acc = accuracy_score(target_test, y_pred)

joblib.dump(model, 'diabetes_model.pkl')