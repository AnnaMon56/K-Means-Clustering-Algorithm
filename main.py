import numpy as np
from davies_bouldin_analysis import TextClusteringAnalysis
from inertia_analysis import InertiaClusterAnalysis
from silhouette_analysis import SilhouetteClusterAnalysis

def load_and_prepare_data(chunk_size: int = 40, sample_size: int = 25000):
    """Load text data and prepare chunks."""
    try:
        with open('text8.txt', 'r', encoding='utf-8') as file:
            text = file.read()
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
            texts = [' '.join(words[i:i+chunk_size])
                     for i in [all_indices[idx] for idx in sampled_indices]]

            print(f"Total possible chunks: {total_chunks}")
            print(f"Analyzing {len(texts)} randomly sampled chunks")
            return texts

    except FileNotFoundError:
        print("Error: text8.txt file not found in the directory.")
        return None
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def run_davies_bouldin_analysis(texts, max_features=2500):
    """Run Davies-Bouldin analysis."""
    print("\n=== Cross-Validated Davies-Bouldin Analysis ===")
    analyzer = TextClusteringAnalysis()

    print("Processing text data...")
    X = analyzer.preprocess_text(texts, max_features=max_features)

    print("Performing cross-validation...")
    cv_results = analyzer.validate_kmeans(X, range(2, 14))

    analyzer.plot_metrics(cv_results)
    optimal_k = analyzer.find_optimal_k(cv_results)
    print(f"Optimal K based on Davies-Bouldin: {optimal_k}")

def run_inertia_analysis(texts, max_features=2500):
    """Run Inertia analysis."""
    print("\n=== Cross-Validated Inertia Analysis ===")
    analyzer = InertiaClusterAnalysis()

    print("Processing text data...")
    X = analyzer.preprocess_text(texts)

    print("Performing cross-validation...")
    cv_results = analyzer.evaluate_kmeans_cv(X, range(2, 14))

    analyzer.plot_metrics(cv_results)
    optimal_k = analyzer.find_optimal_k(cv_results)
    print(f"Optimal K based on Inertia: {optimal_k}")

def run_silhouette_analysis(texts, max_features=2500):
    """Run Silhouette analysis."""
    print("\n=== Cross-Validated Silhouette Analysis ===")
    analyzer = SilhouetteClusterAnalysis()

    print("Processing text data...")
    X = analyzer.preprocess_text(texts)

    print("Performing cross-validation...")
    cv_results = analyzer.evaluate_kmeans_cv(X, range(2, 14))

    analyzer.plot_metrics(cv_results)
    optimal_k = analyzer.find_optimal_k(cv_results)
    print(f"Optimal K based on Silhouette: {optimal_k}")

def main():
    # Load data once for all analyses
    texts = load_and_prepare_data(chunk_size=40, sample_size=25000)
    if texts is None:
        return

    # Run analyses (comment out the ones you don't want to run)
    run_davies_bouldin_analysis(texts)
    # run_inertia_analysis(texts)
    # run_silhouette_analysis(texts)

if __name__ == "__main__":
    main()