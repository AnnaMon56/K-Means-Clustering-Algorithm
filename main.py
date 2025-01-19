from data_processor import DataPreprocessor
from davies_bouldin_analysis import DaviesBouldinEvaluator
from inertia_analysis import InertiaEvaluator
from silhouette_analysis import SilhouetteEvaluator

def run_davies_bouldin_analysis(X):
    print("\n=== Davies-Bouldin Analysis ===")
    evaluator = DaviesBouldinEvaluator()
    results = evaluator.evaluate(X, range(2, 15))
    evaluator.plot_metrics(results)
    optimal_k = evaluator.find_optimal_k(results)
    print(f"Optimal K (Davies-Bouldin): {optimal_k}")
    return optimal_k

def run_inertia_analysis(X):
    print("\n=== Inertia Analysis ===")
    evaluator = InertiaEvaluator()
    results = evaluator.evaluate(X, range(2, 15))
    evaluator.plot_metrics(results)
    optimal_k = evaluator.find_optimal_k(results)
    print(f"Optimal K (Inertia): {optimal_k}")
    return optimal_k

def run_silhouette_analysis(X):
    print("\n=== Silhouette Analysis ===")
    evaluator = SilhouetteEvaluator()
    results = evaluator.evaluate(X, range(2, 15))
    evaluator.plot_metrics(results)
    optimal_k = evaluator.find_optimal_k(results)
    print(f"Optimal K (Silhouette): {optimal_k}")
    return optimal_k

def main():
    # Preprocess data
    preprocessor = DataPreprocessor(max_features=3000)
    texts = preprocessor.load_and_chunk('text8.txt')
    X = preprocessor.preprocess(texts)

    # Run selected analyses
    # Comment out the ones you don't want to use
    run_inertia_analysis(X)
    # run_silhouette_analysis(X)
    # run_davies_bouldin_analysis(X)

if __name__ == "__main__":
    main()