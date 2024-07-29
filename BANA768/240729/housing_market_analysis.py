import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import MNLogit

# Load the data
df = pd.read_csv('HousingMarket.csv')
df_test = pd.read_csv('HousingMarketTest.csv')

# Data preprocessing
def preprocess_data(df):
    # Convert 'bedrooms' to numeric, handling non-numeric values
    df['bedrooms'] = pd.to_numeric(df['bedrooms'].str.extract(r'(\d+)')[0], errors='coerce')
    
    # Convert 'fireplaces' to binary
    df['fireplaces'] = (df['fireplaces'] == 'yes').astype(int)
    
    # Handle numeric columns
    numeric_columns = ['bedrooms', 'averagecostpersquarefootcomparisons', 'cashprice']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill NaN values in numeric columns with median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Handle categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])  # Fill with mode
    
    return df

df = preprocess_data(df)
df_test = preprocess_data(df_test)

# Print data info for debugging
print(df.info())
print(df.head())

# Exploratory Data Analysis
def plot_bedroom_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='bedrooms', data=df)
    plt.title('Distribution of Bedrooms')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Count')
    plt.savefig('bedroom_distribution.png')
    plt.close()

def plot_price_vs_sqft(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='averagecostpersquarefootcomparisons', y='cashprice', data=df)
    plt.title('Price vs Cost per Square Foot')
    plt.xlabel('Average Cost per Square Foot')
    plt.ylabel('Cash Price')
    plt.savefig('price_vs_sqft.png')
    plt.close()

def plot_preference_heatmap(df):
    if 'preference' in df.columns:
        pivot = df.pivot_table(values='preference', index='bedrooms', columns='fireplaces', aggfunc='mean')
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu')
        plt.title('Preference Heatmap: Bedrooms vs Fireplaces')
        plt.savefig('preference_heatmap.png')
        plt.close()
    else:
        print("'preference' column not found in the dataset. Skipping preference heatmap.")

plot_bedroom_distribution(df)
plot_price_vs_sqft(df)
plot_preference_heatmap(df)

# Prepare data for choice modeling
def prepare_choice_data(df):
    if 'preference' in df.columns:
        # Create choice column (1 for chosen option, 0 for others)
        df['choice'] = (df['preference'] == 1).astype(int)
    else:
        print("'preference' column not found. Skipping choice column creation.")
    
    # Create dummy variables for options
    df = pd.get_dummies(df, columns=['option'], prefix='option')
    
    return df

df_choice = prepare_choice_data(df)

# Split data into training and testing sets
train, test = train_test_split(df_choice, test_size=0.2, random_state=42)

# Fit Multinomial Logit model
def fit_mnl_model(train):
    if 'choice' not in train.columns:
        print("'choice' column not found. Cannot fit the model.")
        return None
    
    # Specify the features to use in the model
    features = ['bedrooms', 'fireplaces', 'averagecostpersquarefootcomparisons', 'cashprice', 
                'option_A', 'option_B', 'option_C']
    
    # Ensure all necessary columns are present
    missing_features = [f for f in features if f not in train.columns]
    if missing_features:
        print(f"Missing features: {missing_features}. Cannot fit the model.")
        return None
    
    # Fit the model
    model = MNLogit(train['choice'], train[features])
    results = model.fit()
    
    print(results.summary())
    return results

model_results = fit_mnl_model(train)

# Predict choice shares for test data
def predict_choice_shares(model, test_data):
    if model is None:
        print("Model is not available. Cannot predict choice shares.")
        return None
    
    # Prepare test data
    test_data = prepare_choice_data(test_data)
    
    # Make predictions
    features = ['bedrooms', 'fireplaces', 'averagecostpersquarefootcomparisons', 'cashprice', 
                'option_A', 'option_B', 'option_C']
    
    # Ensure all necessary columns are present
    missing_features = [f for f in features if f not in test_data.columns]
    if missing_features:
        print(f"Missing features in test data: {missing_features}. Cannot predict choice shares.")
        return None
    
    predictions = model.predict(test_data[features])
    
    # Calculate average choice shares
    choice_shares = predictions.mean()
    
    print("Predicted Choice Shares:")
    print(choice_shares)
    
    return choice_shares

choice_shares = predict_choice_shares(model_results, df_test)

# Generate insights
print("\nKey Insights:")
print("1. Most popular number of bedrooms:", df['bedrooms'].mode().values[0])
print("2. Average cash price: $", df['cashprice'].mean())
print("3. Percentage of houses with fireplaces:", (df['fireplaces'] == 1).mean() * 100, "%")

if model_results is not None:
    print("4. Most influential factors in house choice (based on model coefficients):")
    coefficients = model_results.params.abs().sort_values(ascending=False)
    for feature, coef in coefficients.items():
        print(f"   - {feature}: {coef}")

# Save results to a file
with open('housing_market_analysis_results.txt', 'w') as f:
    f.write("Housing Market Analysis Results\n\n")
    if model_results is not None:
        f.write("Model Summary:\n")
        f.write(str(model_results.summary()))
    else:
        f.write("Model could not be fitted due to data issues.\n")
    
    if choice_shares is not None:
        f.write("\n\nPredicted Choice Shares:\n")
        f.write(str(choice_shares))
    else:
        f.write("\n\nChoice shares could not be predicted.\n")
    
    f.write("\n\nKey Insights:\n")
    f.write(f"1. Most popular number of bedrooms: {df['bedrooms'].mode().values[0]}\n")
    f.write(f"2. Average cash price: ${df['cashprice'].mean():.2f}\n")
    f.write(f"3. Percentage of houses with fireplaces: {(df['fireplaces'] == 1).mean() * 100:.2f}%\n")
    
    if model_results is not None:
        f.write("4. Most influential factors in house choice (based on model coefficients):\n")
        for feature, coef in coefficients.items():
            f.write(f"   - {feature}: {coef}\n")