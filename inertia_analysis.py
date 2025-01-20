from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from collections import defaultdict

class InertiaEvaluator:
    def __init__(self, random_state: int = 42, n_splits: int = 5):
        self.random_state = random_state
        self.n_splits = n_splits
        np.random.seed(random_state)

    def evaluate(self, X: np.ndarray, k_range: range) -> Dict:
        results = defaultdict(list)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_indices = list(kf.split(X))

        for k in k_range:
            print(f"Evaluating k={k}")
            fold_scores = []

            for train_idx, val_idx in fold_indices:
                X_train = X[train_idx]
                X_val = X[val_idx]

                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=5,
                    max_iter=200,
                    tol=1e-3,
                    algorithm='elkan'
                )
                kmeans.fit(X_train)
                val_inertia = -kmeans.score(X_val) * X_val.shape[0]
                fold_scores.append(val_inertia)

            results['k_values'].append(k)
            results['inertia_mean'].append(np.mean(fold_scores))
            results['inertia_std'].append(np.std(fold_scores))

        return results

    def plot_metrics(self, results: Dict):
        plt.figure(figsize=(10, 6))
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
        # Find the optimal K based on the elbow method
        # Using a threshold of 10% improvement in inertia
        inertia = np.array(results['inertia_mean'])
        inertia_changes = np.diff(inertia) / inertia[:-1] * 100

        for i, change in enumerate(inertia_changes):
            k = results['k_values'][i+1]
            std = results['inertia_std'][i+1]
            print(f"Improvement from k={k-1} to k={k}: {abs(change):.2f}% ± {std:.2f}")

        threshold = 10
        optimal_idx = 0
        for i, change in enumerate(inertia_changes):
            if abs(change) < threshold:
                optimal_idx = i
                break

        optimal_k = results['k_values'][optimal_idx+1]
        print(f"\nBased on Cross-Validated Elbow Method:")
        print(f"Selected k={optimal_k} as improvements become < {threshold}%")
        return optimal_k