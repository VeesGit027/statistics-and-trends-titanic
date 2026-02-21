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



# 5. Relational Plot (Scatter Plot)
# Relationship between Age and Fare

plt.figure()
plt.scatter(df['Age'], df['Fare'])
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Relationship between Age and Fare")
plt.show()

print("\nRelational Plot Insight:")
print("The scatter plot shows the relationship between passenger age and ticket fare.")
print("There is no strong linear relationship, but higher fares appear more common in certain age groups.")



# 6. Categorical Plot (Bar Chart)
# Survival Count by Sex
plt.figure()
df.groupby('Sex')['Survived'].value_counts().unstack().plot(kind='bar')
plt.title("Survival Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

print("\nCategorical Plot Insight:")
print("The bar chart shows that females had a significantly higher survival rate than males.")

