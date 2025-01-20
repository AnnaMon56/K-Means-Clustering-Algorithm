import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

class DataPreprocessor:
    def __init__(self, max_features: int = 2800):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            min_df=10
        )
        self.scaler = StandardScaler(with_mean=False)

    def load_and_chunk(self, file_path: str, chunk_size: int = 40, sample_size: int = 27000) -> List[str]:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            words = text.split()

            # Initial statistics
            print("\nInitial Dataset Statistics:")
            print(f"Total number of words: {len(words):,}")
            print(f"Number of unique words: {len(set(words)):,}")

            total_chunks = len(words) // chunk_size
            all_indices = range(0, len(words) - chunk_size + 1, chunk_size)

            # Random sampling
            if sample_size < len(all_indices):
                sampled_indices = np.random.choice(len(all_indices), sample_size, replace=False)
            else:
                sampled_indices = range(len(all_indices))

            texts = [' '.join(words[i:i+chunk_size])
                     for i in [all_indices[idx] for idx in sampled_indices]]

            print(f"Total possible chunks: {total_chunks:,}")
            print(f"Analyzing {len(texts):,} randomly sampled chunks")
            return texts

    def preprocess(self, texts: List[str]) -> np.ndarray:
        unlimited_vectorizer = TfidfVectorizer(
            max_features=None,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            min_df=10
        )
        X_unlimited = unlimited_vectorizer.fit_transform(texts)

        X = self.vectorizer.fit_transform(texts)
        X_scaled = self.scaler.fit_transform(X)

        print("\nFeature Statistics:")
        print(f"Number of possible features (with no limit): {X_unlimited.shape[1]:,}")
        print(f"Number of features after limiting to {self.max_features:,}: {X.shape[1]:,}")
        print(f"Final data matrix shape: {X_scaled.shape}")

        return X_scaled.toarray() if hasattr(X_scaled, 'toarray') else X_scaled