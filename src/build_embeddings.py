import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/catalog_prepared.csv")


vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(df["assessment_text"])


with open("data/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("data/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)


with open("data/metadata.pkl", "wb") as f:
    pickle.dump(df.to_dict(orient="records"), f)

print("STEP 5 DONE ")
print("TF-IDF vectors and metadata saved")
