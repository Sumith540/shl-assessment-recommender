from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


with open("data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Query field is required"}), 400

    query = data["query"]

   
    query_vec = vectorizer.transform([query])

  
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

    # Top 5 recommendations
    top_k = min(5, len(metadata))
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = metadata[idx]
        results.append({
            "name": item["assessment_name"],
            "url": item["assessment_url"],
            "test_type": item["test_type"]
        })

    return jsonify({
        "recommended_assessments": results
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
