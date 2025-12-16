import pandas as pd

df = pd.read_csv("data/assessment_catalog.csv")

df["assessment_text"] = (
    df["assessment_name"] + ". " +
    df["description"] + ". Test Type: " +
    df["test_type"]
)

df.to_csv("data/catalog_prepared.csv", index=False)

print("Catalog prepared successfully ")
print(df.head())
