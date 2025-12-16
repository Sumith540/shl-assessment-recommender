import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


queries_df = pd.read_excel("Gen_AI Dataset.xlsx")


with open("data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


def recommend_urls(query, top_k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    urls = []
    for idx in top_indices:
        urls.append(metadata[idx]["assessment_url"])

    return urls


rows = []

for _, row in queries_df.iterrows():
    query = row["Query"]
    recommendations = recommend_urls(query)

    for url in recommendations:
        rows.append({
            "Query": query,
            "Assessment_url": url
        })


output_df = pd.DataFrame(rows)
output_df.to_csv("evaluation/predictions.csv", index=False)

print("STEP 8 DONE ")
print("Saved evaluation/predictions.csv")
print(output_df.head())
