from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from collections import defaultdict

class DaviesBouldinEvaluator:
    def __init__(self, random_state: int = 42, n_splits: int = 5):
        self.random_state = random_state
        self.n_splits = n_splits
        np.random.seed(random_state)

    def evaluate(self, X, k_range: range) -> Dict:
        if hasattr(X, 'toarray'):
            X = X.toarray()
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
                    n_init=5
                )
                kmeans.fit(X_train)
                val_labels = kmeans.predict(X_val)

                if k >= 2:
                    db_score = davies_bouldin_score(X_val, val_labels)
                    fold_scores.append(db_score)

            results['k_values'].append(k)
            if k >= 2:
                results['davies_bouldin_mean'].append(np.mean(fold_scores))
                results['davies_bouldin_std'].append(np.std(fold_scores))
            else:
                results['davies_bouldin_mean'].append(None)
                results['davies_bouldin_std'].append(None)

        return results

    def plot_metrics(self, results: Dict):
        plt.figure(figsize=(10, 6))
        valid_k = [k for k, s in zip(results['k_values'],
                                     results['davies_bouldin_mean']) if s is not None]
        valid_db = [db for db in results['davies_bouldin_mean'] if db is not None]
        valid_std = [std for std in results['davies_bouldin_std'] if std is not None]

        plt.errorbar(valid_k, valid_db, yerr=valid_std, fmt='go-', capsize=5,
                     label=f'{self.n_splits}-fold CV')
        plt.fill_between(valid_k,
                         np.array(valid_db) - np.array(valid_std),
                         np.array(valid_db) + np.array(valid_std),
                         alpha=0.2, color='green')
        plt.title('Cross-Validated Davies-Bouldin Index vs K')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Davies-Bouldin Index')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def find_optimal_k(self, results: Dict) -> int:
        # Find the optimal k based on Davies-Bouldin index
        # Using a thjreshold of 5% improvement
        valid_scores = [s for s in results['davies_bouldin_mean'] if s is not None]
        valid_std = [s for s in results['davies_bouldin_std'] if s is not None]

        improvements = []
        for i in range(len(valid_scores)-1):
            improvement = (valid_scores[i] - valid_scores[i+1]) / valid_scores[i] * 100
            k = i + 3
            print(f"Improvement from k={k-1} to k={k}: {improvement:.2f}% ± {valid_std[i]:.2f}")
            improvements.append(improvement)

        threshold = 5
        optimal_idx = 0
        for i, imp in enumerate(improvements):
            if imp < threshold:
                optimal_idx = i
                break

        optimal_k = optimal_idx + 2
        print(f"\nBased on Occam's Razor principle:")
        print(f"Selected k={optimal_k} as improvements become < {threshold}%")
        return optimal_k