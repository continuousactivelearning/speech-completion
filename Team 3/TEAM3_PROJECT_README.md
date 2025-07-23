# Speech Progress Estimation

This is a research-based project aimed at predicting how much of a speech 
has been completed (in percentage) without relying on the full transcript.  
We estimate this based on partial input - such as semantic features, 
speech structure, and content progression, from live or recorded 
speeches.

---

## Project Objective

To predict the **completion percentage** of a speech chunk by modeling semantic and thematic evolution. The model assumes that meaningful progress isn't linear and requires understanding topic flow and content transitions within the speech.

---

## Approach Summary

We implemented three complementary approaches that contribute to the overall prediction of speech completion:

### 1. Semantic and Structural Feature Modeling – *Ananya*

- Transcripts were divided into semantic chunks.
- Sentence embeddings (using SentenceTransformer MiniLM) were used to compute semantic similarity between adjacent chunks.
- **Novelty** was measured as `1 - cosine_similarity`, capturing how much each chunk deviates from the previous.
- **Change point detection** was applied using the PELT algorithm to detect structural shifts.
- Final feature vector included:
  - Mean, variance, and trend of novelty
  - Number of change points
  - Change point density and recency
- **Model**: Random Forest Regressor with GridSearchCV for hyperparameter tuning.

### 2. Topic Modeling-Based Completion Estimation – *Sailakshmi*

- Each transcript was divided into 3-sentence chunks and treated as mini-documents.
- **BERTopic** (BERT + UMAP + HDBSCAN) was used to assign topic labels to each chunk.
- Cumulative unique topics (`Uᵢ`) were tracked across the speech.
- **Saturation point** was detected as the index where new topic introductions plateau.
- Estimated completion: `(Saturation Chunk Index / Total Chunks) × 100`.
- **Final Model**: Random Forest Regressor using saturation metrics as features.

### 3. Ensemble Learning and Feature Fusion – *Aryan*

- Input data: `{Speech_ID, Chunk_ID, Chunk Text, Completion %}`
- Extracted:
  - **Textual features**: word/char counts, punctuation ratios
  - **Semantic embeddings**: SentenceTransformer (MPNet, 768-dim)
  - **Clustering**: PCA → UMAP → HDBSCAN for thematic patterns
  - **Temporal patterns**: recurrence of clusters, ratio of new themes
- **Model**: LightGBM with early stopping and validation monitoring.
- Final model combined all features into a unified 17-dimensional vector per chunk.
- Evaluation Metric: Mean Absolute Error (MAE)

---

## Author Contributions

| Name            | Contributions                                                                 |
|------------------|--------------------------------------------------------------------------------|
| **Ananya Thakur** | Semantic similarity modeling, change point detection, feature engineering, full-stack dashboard, ensemble logic |
| **Aryan**         | Feature extraction (semantic + temporal), clustering pipeline, LightGBM ensemble model |
| **Sailakshmi**    | Topic modeling using BERTopic, saturation-based completion estimation         |

---
