import pandas as pd

# Load the data from CSV
df = pd.read_csv('data.csv')

# Display the DataFrame
print("\nDataFrame Shape:", df.shape)
print("\nFirst few rows of the data:")
print(df.head())
print("\nDataFrame Info:")
print(df.info())
