import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
num_students = 500

# Feature generators
def generate_hours_studied():
    return np.clip(np.random.normal(4, 2), 0, 10)

def generate_attendance():
    return np.clip(np.random.normal(85, 10), 50, 100)

def generate_parental_education():
    return random.choice(['HighSchool', 'Bachelors', 'Masters', 'PhD'])

def generate_extracurricular():
    return random.choice(['Yes', 'No'])

def generate_sleep_hours():
    return np.clip(np.random.normal(7, 1.5), 3, 10)

def generate_internet_access():
    return random.choice(['Yes', 'No'])

def calculate_performance(hours, attendance, sleep, edu, extra, net):
    score = hours * 3+ attendance * 0.50+ sleep * 1
    score += 5 if edu in ['Masters', 'PhD'] else 0
    score += 3 if extra == 'Yes' else 0
    score -= 2 if net == 'Yes' else 0

    if score > 80:
        return 'High'
    elif score > 70:
        return 'Medium'
    else:
        return 'low'

# Generate dataset
data = []

for _ in range(num_students):
    hours = generate_hours_studied()
    attendance = generate_attendance()
    sleep = generate_sleep_hours()
    edu = generate_parental_education()
    extra = generate_extracurricular()
    net = generate_internet_access()
    performance = calculate_performance(hours, attendance, sleep, edu, extra, net)
    
    data.append([
        round(hours, 2),
        round(attendance, 2),
        sleep,
        edu,
        extra,
        net,
        performance
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    'hours_studied', 'attendance_percentage', 'sleep_hours',
    'parental_education_level', 'extracurricular_participation',
    'internet_access', 'performance'
])

# Save to CSV
df.to_csv('student_performance_dataset.csv', index=False)
print("Dataset generated and saved as 'student_performance_dataset.csv'")
