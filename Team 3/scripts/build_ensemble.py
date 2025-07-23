import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import umap
import hdbscan
from hdbscan import approximate_predict
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

# === Custom Transformers ===

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        features = []
        for text in X:
            words = text.split()
            num_words = len(words)
            num_chars = len(text)
            num_commas = text.count(',')
            num_periods = text.count('.')
            num_exclaims = text.count('!')
            num_questions = text.count('?')
            unique_words = len(set(words))
            frac_unique = unique_words / (num_words + 1e-5)
            features.append([
                num_words, num_chars, num_commas, num_periods,
                num_exclaims, num_questions, unique_words, frac_unique
            ])
        return np.array(features)
    def get_feature_names_out(self, input_features=None):
        return ['num_words', 'num_chars', 'num_commas', 'num_periods',
                'num_exclaims', 'num_questions', 'unique_words', 'fraction_unique_words']

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = None
    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name)
        return self
    def transform(self, X):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        # Accept both list or pandas Series
        texts = X.tolist() if hasattr(X, 'tolist') else list(X)
        return self.model.encode(texts, show_progress_bar=False)

class ClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_pca_components=5, n_umap_components=2):
        self.n_pca_components = n_pca_components
        self.n_umap_components = n_umap_components
        self.pca = PCA(n_components=n_pca_components, random_state=42)
        self.umap_model = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=n_umap_components, random_state=42)
        self.clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    def fit(self, X, y=None):
        pca_features = self.pca.fit_transform(X)
        umap_features = self.umap_model.fit_transform(pca_features)
        self.clusterer.fit(umap_features)
        return self
    def transform(self, X):
        pca_features = self.pca.transform(X)
        umap_features = self.umap_model.transform(pca_features)
        cluster_labels, _ = approximate_predict(self.clusterer, umap_features)
        is_noise = (cluster_labels == -1).astype(int)
        return np.column_stack([pca_features, cluster_labels, is_noise]), cluster_labels
    def get_feature_names_out(self, input_features=None):
        return [f'pca_{i+1}' for i in range(self.n_pca_components)] + ['cluster', 'is_noise']

# === Main Predictor ===

class SpeechCompletionPredictor(BaseEstimator):
    def __init__(self):
        self.text_extractor = TextFeatureExtractor()
        self.embedding_transformer = EmbeddingTransformer()
        self.clustering_transformer = ClusteringTransformer()
        self.lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
        self.feature_names = None
        self.is_fitted = False

    def _compute_temporal_features(self, df_temp):
        cluster_progress = []
        fraction_unique_clusters = []
        for _, group in df_temp.groupby("Speech_ID"):
            seen = set()
            total_clusters = set(group['cluster'].unique())
            cluster_seen = []
            frac_unique = []
            for c in group['cluster']:
                cluster_seen.append(int(c in seen))
                seen.add(c)
                frac_unique.append(len(seen) / len(total_clusters))
            cluster_progress.extend(cluster_seen)
            fraction_unique_clusters.extend(frac_unique)
        return np.array(cluster_progress), np.array(fraction_unique_clusters)

    def _prepare_features(self, X_df):
        text_features = self.text_extractor.transform(X_df['Chunk'])
        embeddings = self.embedding_transformer.transform(X_df['Chunk'])
        clustering_features, cluster_labels = self.clustering_transformer.transform(embeddings)

        df_temp = X_df.copy()
        df_temp['cluster'] = cluster_labels
        cluster_seen, frac_unique = self._compute_temporal_features(df_temp)

        all_features = np.column_stack([
            text_features,
            clustering_features,
            cluster_seen.reshape(-1, 1),
            frac_unique.reshape(-1, 1)
        ])
        return all_features

    def fit(self, X_df, y, eval_set=None):
        text_features = self.text_extractor.fit_transform(X_df['Chunk'])
        embeddings = self.embedding_transformer.fit_transform(X_df['Chunk'])
        clustering_features, cluster_labels = self.clustering_transformer.fit_transform(embeddings)

        df_temp = X_df.copy()
        df_temp['cluster'] = cluster_labels
        cluster_seen, frac_unique = self._compute_temporal_features(df_temp)

        all_features = np.column_stack([
            text_features,
            clustering_features,
            cluster_seen.reshape(-1, 1),
            frac_unique.reshape(-1, 1)
        ])

        self.feature_names = (
            self.text_extractor.get_feature_names_out() +
            self.clustering_transformer.get_feature_names_out() +
            ['cluster_seen_before', 'fraction_unique_clusters']
        )

        if eval_set is not None:
            X_val_df, y_val = eval_set
            val_features = self._prepare_features(X_val_df)
            self.lgb_model.fit(
                all_features, y,
                eval_set=[(val_features, y_val)],
                eval_metric='mae',
                callbacks=[early_stopping(50), log_evaluation(50)]
            )
        else:
            self.lgb_model.fit(all_features, y)

        self.is_fitted = True
        return self

    def predict(self, X_df):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        features = self._prepare_features(X_df)
        return self.lgb_model.predict(features)

    def get_feature_importance(self):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.lgb_model.feature_importances_ / self.lgb_model.feature_importances_.sum()
        }).sort_values(by='importance', ascending=False)
