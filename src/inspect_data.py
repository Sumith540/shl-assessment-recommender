import pandas as pd

# Load dataset
df = pd.read_excel("Gen_AI Dataset.xlsx")

print("Total rows:", df.shape[0])
print("\nColumns:")
for col in df.columns:
    print("-", col)

print("\nSample data:")
print(df.head(3))
