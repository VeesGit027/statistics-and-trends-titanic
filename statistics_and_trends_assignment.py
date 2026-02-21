# **Statistics and Trends Assignment**
# Titanic Dataset Analysis
"""

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# 2. Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# 3. Data Cleaning
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Remove duplicates (if any)
df.drop_duplicates(inplace=True)


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 4. Data Preparation
# Convert Survived to categorical for better plotting
df['Survived'] = df['Survived'].astype('category')

# Create age groups for categorical analysis
df['Age_Group'] = pd.cut(df['Age'],
                         bins=[0, 12, 18, 35, 60, 100],
                         labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
