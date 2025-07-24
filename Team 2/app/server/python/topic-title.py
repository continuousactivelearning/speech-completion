import pandas as pd
import spacy
import json
from pathlib import Path
import argparse

from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from top2vec import Top2Vec

parser = argparse.ArgumentParser(description="Extract topic titles using various algorithms")
parser.add_argument("--input", required=True, help="Path to input CSV")
parser.add_argument("--output", required=True, help="Path to output JSON")
args = parser.parse_args()

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv(args.input)
df = df[df["text"].notnull() & df["cluster"].notnull()]

grouped = df.groupby("cluster")["text"].apply(lambda x: " ".join(x)).reset_index()
grouped.columns = ["cluster", "combined_text"]

# BERTopic
bertopic_model = BERTopic()
bertopic_topics, _ = bertopic_model.fit_transform(grouped["combined_text"])

# LDA function
def apply_lda(texts, num_topics=1, num_words=5):
    tokenized = [[token.text.lower() for token in nlp(text) if token.is_alpha and not token.is_stop] for text in texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return [[word for word, prob in lda_model.show_topic(i, topn=num_words)] for i in range(num_topics)]

# TextRank-like
def textrank_keywords(text, top_n=5):
    doc = nlp(text)
    candidates = [chunk.text for chunk in doc.noun_chunks]
    if not candidates:
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(candidates)
    scores = X.sum(axis=0).A1
    keywords = [vectorizer.get_feature_names_out()[i] for i in scores.argsort()[::-1][:top_n]]
    return keywords

# Top2Vec
top2vec_model = Top2Vec(grouped["combined_text"].tolist(), speed="learn", workers=4)

output = []

for i, row in grouped.iterrows():
    cluster_num = int(row["cluster"])
    combined_text = row["combined_text"]

    # BERTopic
    bertopic_words = [
        word for word, _ in bertopic_model.get_topic(bertopic_topics[i])
    ]

    # LDA
    lda_words = apply_lda([combined_text])[0]

    # TextRank
    textrank_words = textrank_keywords(combined_text)

    # Top2Vec
    try:
        topic_words, _ = top2vec_model.get_topics(1)
        top2vec_words = topic_words[0][:5]  # Top 5 words
    except Exception as e:
        top2vec_words = []

    cluster_output = {
        "cluster": cluster_num,
        "text_rank": textrank_words,
        "lda": lda_words,
        "bertopic": bertopic_words,
        "top2vec": top2vec_words,
    }

    output.append(cluster_output)

Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))

print(f"OK: saved {args.output} with {len(output)} clusters")