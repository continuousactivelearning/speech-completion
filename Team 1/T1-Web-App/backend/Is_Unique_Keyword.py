from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def is_unique(keyword: str, existing_words: list[str], threshold=0.8):
    if not existing_words: 
        return True

    keyword_emb = embedding_model.encode([keyword])
    existing_embeddings = embedding_model.encode(existing_words)

    sims = cosine_similarity(keyword_emb, existing_embeddings)
    return np.max(sims) < threshold


