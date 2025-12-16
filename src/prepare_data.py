import pandas as pd

# Load dataset
df = pd.read_excel("Gen_AI Dataset.xlsx")


NAME_COL = "Assessment Name"
DESC_COL = "Description"
TYPE_COL = "Test Type"
URL_COL = "URL"

def build_text(row):
    text = f"""
    Assessment Name: {row.get(NAME_COL, '')}
    Description: {row.get(DESC_COL, '')}
    Test Type: {row.get(TYPE_COL, '')}
    """
    return text.strip()

df["assessment_text"] = df.apply(build_text, axis=1)


final_df = df[[NAME_COL, URL_COL, TYPE_COL, "assessment_text"]]


final_df.to_csv("data/assessments_cleaned.csv", index=False)

print("STEP 3 DONE ")
print("Saved to data/assessments_cleaned.csv")
print(final_df.head(2))
