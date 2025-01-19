import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import List, Dict
from collections import defaultdict

class InertiaClusterAnalysis:
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

                # Calculate inertia on validation set
                val_labels = kmeans.predict(X_val)
                val_centers = kmeans.cluster_centers_

                # Calculate inertia manually for validation set
                val_inertia = 0
                for i, label in enumerate(val_labels):
                    val_inertia += np.sum((X_val[i] - val_centers[label]) ** 2)

                fold_scores.append(val_inertia)

            results['k_values'].append(k)
            results['inertia_mean'].append(np.mean(fold_scores))
            results['inertia_std'].append(np.std(fold_scores))

        return results

    def plot_metrics(self, results: Dict):
        plt.figure(figsize=(10, 6))

        # Plot inertia with error bars
        plt.errorbar(results['k_values'],
                     results['inertia_mean'],
                     yerr=results['inertia_std'],
                     fmt='bo-',
                     capsize=5,
                     label=f'{self.n_splits}-fold CV')

        plt.fill_between(results['k_values'],
                         np.array(results['inertia_mean']) - np.array(results['inertia_std']),
                         np.array(results['inertia_mean']) + np.array(results['inertia_std']),
                         alpha=0.2, color='blue')

        plt.title('Cross-Validated Elbow Method: Inertia vs K')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def find_optimal_k(self, results: Dict) -> int:
        """Find optimal k using elbow method with cross-validation"""
        inertia = np.array(results['inertia_mean'])

        # Calculate percentage changes
        inertia_changes = np.diff(inertia) / inertia[:-1] * 100

        # Print improvements with standard deviations
        for i, change in enumerate(inertia_changes):
            k = results['k_values'][i+1]
            std = results['inertia_std'][i+1]
            print(f"Improvement from k={k-1} to k={k}: {abs(change):.2f}% ± {std:.2f}")

        # Find elbow point using percentage change threshold
        threshold = 10  # 10% improvement threshold
        optimal_idx = 0
        for i, change in enumerate(inertia_changes):
            if abs(change) < threshold:
                optimal_idx = i
                break

        optimal_k = results['k_values'][optimal_idx+1]
        print(f"\nBased on Cross-Validated Elbow Method:")
        print(f"Selected k={optimal_k} as improvements become < {threshold}%")
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
    analyzer = InertiaClusterAnalysis(n_splits=5)
    print("\n=== Cross-Validated Inertia-based Analysis ===")

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