import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


with open("data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


def recommend_assessments(query, top_k=5):
    
    query_vec = vectorizer.transform([query])

    
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]


    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Prepare results
    results = []
    for idx in top_indices:
        item = metadata[idx]
        results.append({
            "assessment_name": item["assessment_name"],
            "assessment_url": item["assessment_url"],
            "test_type": item["test_type"],
            "score": float(similarities[idx])
        })

    return results


# Simple test
if __name__ == "__main__":
    query = "Looking for a Java developer with good problem solving skills"
    recommendations = recommend_assessments(query)

    print("Query:", query)
    print("\nRecommended Assessments:")
    for r in recommendations:
        print("-", r["assessment_name"], "|", r["assessment_url"])
