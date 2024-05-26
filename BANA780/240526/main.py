import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset from seaborn library
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
titanic.head()

# Check for missing values
titanic.isnull().sum()

# Fill missing age values with the median age
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Fill missing embarked values with the mode
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Drop rows with missing values in 'deck' and 'embark_town'
titanic.drop(columns=['deck', 'embark_town'], inplace=True)

# Fill missing 'embarked' values with mode
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Display the cleaned dataset
titanic.isnull().sum()

# Plot common values for each variable
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot survival
sns.countplot(data=titanic, x='survived', ax=axs[0, 0])
axs[0, 0].set_title('Survival Status')

# Plot class
sns.countplot(data=titanic, x='pclass', ax=axs[0, 1])
axs[0, 1].set_title('Passenger Class')

# Plot gender
sns.countplot(data=titanic, x='sex', ax=axs[1, 0])
axs[1, 0].set_title('Gender')

# Plot embarked
sns.countplot(data=titanic, x='embarked', ax=axs[1, 1])
axs[1, 1].set_title('Embarked Location')

plt.tight_layout()
plt.show()

# Plot survival by class
plt.figure(figsize=(10, 6))
sns.countplot(data=titanic, x='pclass', hue='survived')
plt.title('Survival by Class')
plt.show()

# Plot survival by gender
plt.figure(figsize=(10, 6))
sns.countplot(data=titanic, x='sex', hue='survived')
plt.title('Survival by Gender')
plt.show()

# Plot survival by age
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='age', hue='survived', multiple='stack', kde=False)
plt.title('Survival by Age')
plt.show()

# Create age groups
titanic['age_group'] = pd.cut(titanic['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80], labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])

# Plot survival by age group and class
plt.figure(figsize=(12, 8))
sns.countplot(data=titanic, x='age_group', hue='survived', palette='pastel', col='pclass')
plt.title('Survival by Age Group and Class')
plt.show()

# Plot survival by embarkation point and class
plt.figure(figsize=(12, 8))
sns.countplot(data=titanic, x='embarked', hue='survived', palette='pastel', col='pclass')
plt.title('Survival by Embarkation Point and Class')
plt.show()
