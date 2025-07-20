import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')

# Features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with hyperparameter tuning
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}

rfc = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rfc, param_distributions=params, n_iter=10, scoring='accuracy', cv=5, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)

# Best model
model = search.best_estimator_

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Improved Model Accuracy: {:.2f}%".format(accuracy * 100))

# Save model
joblib.dump(model, 'diabetes_model.pkl')

