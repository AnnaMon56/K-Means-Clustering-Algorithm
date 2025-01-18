import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class TextClusteringAnalysis:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=4500,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        self.scaler = StandardScaler(with_mean=False)

    def preprocess_text(self, texts: List[str]) -> np.ndarray:
        #Transform text data into TF-IDF features and scale them.
        X = self.vectorizer.fit_transform(texts)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def evaluate_kmeans(self, X: np.ndarray, k_range: range) -> Dict:
        results = {
            'k_values': [],
            'davies_bouldin': [] # average similarity measure of each cluster with the cluster most similar to it. Lower is better
        }

        X_dense = X.toarray()

        for k in k_range:
            print(f"Evaluating k={k}")
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(X)

            # Calculate metrics
            results['k_values'].append(k)
            if k >= 2:
                results['davies_bouldin'].append(davies_bouldin_score(X_dense, labels))
            else:
                results['davies_bouldin'].append(None)

        return results

    def plot_metrics(self, results: Dict):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # Single plot with adjusted size

        # Plot Davies-Bouldin index
        valid_k = [k for k, s in zip(results['k_values'], results['davies_bouldin']) if s is not None]
        valid_db = [db for db in results['davies_bouldin'] if db is not None]
        ax.plot(valid_k, valid_db, 'go-')
        ax.set_title('Davies-Bouldin Index vs K')
        ax.set_xlabel('Number of clusters (K)')
        ax.set_ylabel('Davies-Bouldin Index')

        plt.tight_layout()
        plt.show()

    def find_optimal_k(self, results: Dict) -> int:
        # suggest an optimal k based on davies bouldin score
        db_scores = np.array(results['davies_bouldin'])
        db_min_idx = np.argmin(db_scores[1:]) + 1
        optimal_k = results['k_values'][db_min_idx]
        return optimal_k

def main():
    try:
        with open('text8.txt', 'r', encoding='utf-8') as file:
            text = file.read()
            # Split the text into sentences or chunks
            words = text.split()
            chunk_size = 40  # 50 words per chunk
            texts = [' '.join(words[i:i+chunk_size])
                     for i in range(0, len(words), chunk_size)
                     if i+chunk_size < len(words)]
            # Limiting the number of chunks to analyze
            # File is too big to analyze all chunks
            texts = texts[:40000]

            print(f"Number of text chunks to analyze: {len(texts)}")
    except FileNotFoundError:
        print("Error: text8.txt file not found in the directory.")
        return
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return

    # Initialize analyzer
    analyzer = TextClusteringAnalysis()

    print("Processing text data...")
    # Preprocess text data
    X = analyzer.preprocess_text(texts)

    print("Evaluate KMeans clustering...")
    # Evaluate different K values
    results = analyzer.evaluate_kmeans(X, range(2, 18))  # Test K from 2 to 20
    # Plot results
    analyzer.plot_metrics(results)

    # Find suggested optimal K
    optimal_k = analyzer.find_optimal_k(results)
    print(f"Optimal number of clusters: {optimal_k}")

if __name__ == "__main__":
    print("Starting execution...")
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")