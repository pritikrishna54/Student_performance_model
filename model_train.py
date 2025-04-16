import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("student_performance_dataset.csv")

# Encode categorical features
label_encoders = {}
categorical_cols = ['parental_education_level', 'extracurricular_participation', 'internet_access', 'performance']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop('performance', axis=1)
y = df['performance']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, 'student_performance_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Model and encoders saved to 'student_performance_model.pkl' and 'label_encoders.pkl'")
