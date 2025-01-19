import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import List, Dict
from collections import defaultdict

class SilhouetteClusterAnalysis:
    def __init__(self, random_state: int = 42, n_splits: int = 5):
        self.random_state = random_state
        self.n_splits = n_splits
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        self.scaler = StandardScaler(with_mean=False)

    def preprocess_text(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.fit_transform(texts)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def evaluate_kmeans_cv(self, X: np.ndarray, k_range: range) -> Dict:
        results = defaultdict(list)
        X_dense = X.toarray()

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for k in k_range:
            print(f"Evaluating k={k}")
            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_dense)):
                X_train = X_dense[train_idx]
                X_val = X_dense[val_idx]

                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=10
                )
                kmeans.fit(X_train)
                val_labels = kmeans.predict(X_val)

                if k >= 2:
                    silhouette = silhouette_score(X_val, val_labels)
                    fold_scores.append(silhouette)

            results['k_values'].append(k)
            if k >= 2:
                results['silhouette_mean'].append(np.mean(fold_scores))
                results['silhouette_std'].append(np.std(fold_scores))
            else:
                results['silhouette_mean'].append(None)
                results['silhouette_std'].append(None)

        return results

    def plot_metrics(self, results: Dict):
        plt.figure(figsize=(10, 6))

        # Get valid k values and scores
        valid_k = [k for k, s in zip(results['k_values'],
                                     results['silhouette_mean']) if s is not None]
        valid_scores = [s for s in results['silhouette_mean'] if s is not None]
        valid_std = [s for s in results['silhouette_std'] if s is not None]

        # Plot with error bars
        plt.errorbar(valid_k, valid_scores,
                     yerr=valid_std,
                     fmt='ro-',
                     capsize=5,
                     label=f'{self.n_splits}-fold CV')

        plt.fill_between(valid_k,
                         np.array(valid_scores) - np.array(valid_std),
                         np.array(valid_scores) + np.array(valid_std),
                         alpha=0.2, color='red')

        plt.title('Cross-Validated Silhouette Score vs K')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def find_optimal_k(self, results: Dict) -> int:
        """Find optimal k using silhouette score with cross-validation"""
        valid_scores = [s for s in results['silhouette_mean'] if s is not None]
        valid_std = [s for s in results['silhouette_std'] if s is not None]
        optimal_idx = np.argmax(valid_scores)
        optimal_k = results['k_values'][optimal_idx+1]  # +1 because k starts at 2

        # Print scores with standard deviations
        for i, (score, std) in enumerate(zip(valid_scores, valid_std)):
            k = i + 2  # Since silhouette starts from k=2
            print(f"K={k}: Silhouette Score = {score:.4f} ± {std:.4f}")

        print(f"\nOptimal K based on highest Cross-Validated Silhouette Score: {optimal_k}")
        return optimal_k

def main():
    try:
        with open('text8.txt', 'r', encoding='utf-8') as file:
            text = file.read()
            words = text.split()
            chunk_size = 40
            texts = [' '.join(words[i:i+chunk_size])
                     for i in range(0, len(words), chunk_size)
                     if i+chunk_size < len(words)]
            texts = texts[:10000]
            print(f"Number of text chunks to analyze: {len(texts)}")
    except FileNotFoundError:
        print("Error: text8.txt file not found in the directory.")
        return
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return

    # Initialize analyzer with cross-validation
    analyzer = SilhouetteClusterAnalysis(n_splits=5)
    print("\n=== Cross-Validated Silhouette-based Analysis ===")

    # Preprocess and evaluate
    X = analyzer.preprocess_text(texts)
    results = analyzer.evaluate_kmeans_cv(X, range(2, 18))

    # Plot and find optimal K
    analyzer.plot_metrics(results)
    optimal_k = analyzer.find_optimal_k(results)

if __name__ == "__main__":
    print("Starting execution...")
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")