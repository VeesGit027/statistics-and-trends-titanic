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

"""# **3. Data Cleaning**"""

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

"""# **4. Data Preparation**"""

# Convert Survived to categorical for better plotting
df['Survived'] = df['Survived'].astype('category')

# Create age groups for categorical analysis
df['Age_Group'] = pd.cut(df['Age'],
                         bins=[0, 12, 18, 35, 60, 100],
                         labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

"""# **5. Relational Plot (Scatter Plot)**"""

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

"""# **6. Categorical Plot (Bar Chart)**"""

# Survival Count by Sex
plt.figure()
df.groupby('Sex')['Survived'].value_counts().unstack().plot(kind='bar')
plt.title("Survival Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

print("\nCategorical Plot Insight:")
print("The bar chart shows that females had a significantly higher survival rate than males.")

"""# **7. Statistical Plot (Correlation Heatmap)**"""

plt.figure()
numeric_df = df.select_dtypes(include=np.number)
correlation = numeric_df.corr()

sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap of Numerical Variables")
plt.show()

print("\nStatistical Plot Insight:")
print("The heatmap shows correlations between numerical variables.")
print("Fare has moderate correlation with Pclass, and Age shows weak correlations overall.")

"""# **8. Four Main Statistical Moments (Using Fare)**"""

fare_mean = df['Fare'].mean()
fare_variance = df['Fare'].var()
fare_skewness = skew(df['Fare'])
fare_kurtosis = kurtosis(df['Fare'])

print("\nStatistical Moments for Fare:")
print(f"Mean: {fare_mean}")
print(f"Variance: {fare_variance}")
print(f"Skewness: {fare_skewness}")
print(f"Kurtosis: {fare_kurtosis}")

print("\nStatistical Interpretation:")
print("Mean represents the average fare paid by passengers.")
print("Variance indicates how spread out the fare values are.")
print("Positive skewness suggests a right-skewed distribution (some very high fares).")
print("Positive kurtosis indicates heavy tails compared to a normal distribution.")

"""# **9. Critical Assessment**"""

print("\nCritical Assessment:")
print("The dataset contains missing values and imbalances (e.g., gender distribution).")
print("Dropping the Cabin column may remove potentially useful information.")
print("Outliers in Fare significantly affect skewness and kurtosis.")
print("Results are limited to passengers recorded and may not represent all individuals aboard.")