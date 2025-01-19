import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import List, Dict
from collections import defaultdict

class TextClusteringAnalysis:
    def __init__(self, random_state: int = 42, n_splits: int = 5):
        self.random_state = random_state
        self.n_splits = n_splits
        np.random.seed(random_state)

    def create_chunks(self, text: str, chunk_size: int = 40, sample_size: int = 12000) -> List[str]:
        """Create chunks and sample randomly."""
        words = text.split()
        total_chunks = len(words) // chunk_size

        # Create all possible chunk indices
        all_indices = range(0, len(words) - chunk_size + 1, chunk_size)

        # Randomly sample indices
        if sample_size < len(all_indices):
            sampled_indices = np.random.choice(len(all_indices), sample_size, replace=False)
        else:
            sampled_indices = range(len(all_indices))

        # Create chunks from sampled indices
        chunks = [' '.join(words[i:i+chunk_size])
                  for i in [all_indices[idx] for idx in sampled_indices]]

        print(f"Total possible chunks: {total_chunks}")
        print(f"Analyzing {len(chunks)} randomly sampled chunks")
        return chunks

    def preprocess_text(self, texts: List[str], max_features: int = 1800) -> np.ndarray:
        """Transform text data into TF-IDF features and scale them."""
        print(f"Vectorizing with {max_features} features...")

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            min_df=10  # Increased min_df to reduce features
        )

        X = self.vectorizer.fit_transform(texts)
        print(f"Actual number of features: {X.shape[1]}")

        self.scaler = StandardScaler(with_mean=False)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled.toarray()  # Convert to dense once

    def validate_kmeans(self, X: np.ndarray, k_range: range) -> Dict:
        """Cross-validate the clustering results."""
        results = defaultdict(list)

        # Pre-compute fold indices
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_indices = list(kf.split(X))

        for k in k_range:
            print(f"Cross-validating k={k}")
            fold_scores = []

            for train_idx, val_idx in fold_indices:
                X_train = X[train_idx]
                X_val = X[val_idx]

                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=5,  # Reduced from 10
                    max_iter=200,
                    tol=1e-3,  # Increased tolerance
                    algorithm='elkan'  # Generally faster for k-means
                )
                kmeans.fit(X_train)
                val_labels = kmeans.predict(X_val)

                if k >= 2:
                    db_score = davies_bouldin_score(X_val, val_labels)
                    fold_scores.append(db_score)

            results['k_values'].append(k)
            if k >= 2:
                mean_score = np.mean(fold_scores)
                std_score = np.std(fold_scores)
                results['davies_bouldin_mean'].append(mean_score)
                results['davies_bouldin_std'].append(std_score)
                print(f"K={k}: DB Score = {mean_score:.4f} ± {std_score:.4f}")
            else:
                results['davies_bouldin_mean'].append(None)
                results['davies_bouldin_std'].append(None)

        return results

    def plot_metrics(self, results: Dict):
        """Plot cross-validation results."""
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
        """Find optimal k and calculate improvements between successive K values."""
        valid_scores = [s for s in results['davies_bouldin_mean'] if s is not None]
        valid_std = [s for s in results['davies_bouldin_std'] if s is not None]

        # Calculate percentage improvements
        improvements = []
        for i in range(len(valid_scores)-1):
            improvement = (valid_scores[i] - valid_scores[i+1]) / valid_scores[i] * 100
            k = i + 3  # Since we start from k=2
            print(f"Improvement from k={k-1} to k={k}: {improvement:.2f}% ± {valid_std[i]:.2f}")
            improvements.append(improvement)

        # Find point where improvements become minimal
        threshold = 5  # 5% improvement threshold
        optimal_idx = 0
        for i, imp in enumerate(improvements):
            if imp < threshold:
                optimal_idx = i
                break

        optimal_k = optimal_idx + 2  # +2 because we start from k=2
        print(f"\nBased on Occam's Razor principle:")
        print(f"Selected k={optimal_k} as improvements become < {threshold}%")
        return optimal_k

def main():
    try:
        with open('text8.txt', 'r', encoding='utf-8') as file:
            text = file.read()

            # Initialize analyzer
            analyzer = TextClusteringAnalysis()

            # Create chunks with sampling
            texts = analyzer.create_chunks(text, chunk_size=40, sample_size=25000)

            print("Processing text data...")
            X = analyzer.preprocess_text(texts, max_features=2500)

            # Perform cross-validation
            print("Performing cross-validation...")
            cv_results = analyzer.validate_kmeans(X, range(2, 14))  # K from 2 to 10

            # Plot results
            analyzer.plot_metrics(cv_results)

            # Find optimal K with improvements analysis
            optimal_k = analyzer.find_optimal_k(cv_results)
            print(f"Optimal K based on cross-validation: {optimal_k}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    print("Starting execution...")
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")