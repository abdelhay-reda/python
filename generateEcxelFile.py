import pandas as pd
import numpy as np

# Generate sample data for the Heart Disease Dataset
# Each row represents a patient with various features and the target variable (presence of heart disease)

# Define the number of samples
num_samples = 1000

# Define the features
age = np.random.randint(29, 77, num_samples)
sex = np.random.choice([0, 1], size=num_samples)
cp = np.random.randint(0, 4, num_samples)
trestbps = np.random.randint(94, 200, num_samples)
chol = np.random.randint(126, 564, num_samples)
fbs = np.random.choice([0, 1], size=num_samples)
restecg = np.random.randint(0, 3, num_samples)
thalach = np.random.randint(71, 202, num_samples)
exang = np.random.choice([0, 1], size=num_samples)
oldpeak = np.random.uniform(0, 6.2, num_samples)
slope = np.random.randint(0, 3, num_samples)
ca = np.random.randint(0, 5, num_samples)
thal = np.random.randint(0, 4, num_samples)

# Define the target variable (presence of heart disease)
target = np.random.choice([0, 1], size=num_samples)

# Create a DataFrame
data = pd.DataFrame({
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal,
    'target': target
})

# Write the data to an Excel file
data.to_excel('generatedFiles/heart_disease_dataset.xlsx', index=False)
