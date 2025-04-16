import pandas as pd
import numpy as np


np.random.seed(42)
num_students = 100

data = {
    "Study_Hours": np.random.randint(1, 10, num_students),
    "Class_Participation": np.random.randint(1, 10, num_students),
    "Assignment_Score": np.random.randint(40, 100, num_students),
    "Performance": np.random.choice(["Pass", "Fail"], num_students, p=[0.7, 0.3])
}

# Create DataFrame
dataset = pd.DataFrame(data)
print(dataset.head())

# Save to CSV
dataset.to_csv("student_performance.csv", index=False)
print("Dataset saved to 'student_performance.csv'")