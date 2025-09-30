from data import load_and_preprocess_data
from model import NCF
from neumf_model import NeuralMatrixFactorization, NeuMF_Shared
from train import train_model
from evaluate import evaluate_model
from recommendation_system import create_production_system
from model_comparison import run_model_comparison, run_architecture_comparison
from cold_start import ColdStartHandler

def main():
    # Config
    file_path = 'ecommerce_data.csv'
    implicit = True
    
    print("=== LOADING AND PREPROCESSING DATA ===")
    train_loader, val_loader, test_loader, num_users, num_items = load_and_preprocess_data(
        file_path, implicit=implicit
    )
    
    print("\n=== TRAINING NeuMF MODEL ===")
    print("Comparing 3 architectures:")
    print("1. Original NCF (MLP-only)")
    print("2. NeuMF with separate embeddings (GMF + MLP)")
    print("3. NeuMF with shared embeddings (GMF + MLP)")
    
    # Train all 3 models for comparison
    print("\n--- Training Original NCF ---")
    ncf_model = NCF(num_users, num_items)
    trained_ncf = train_model(ncf_model, train_loader, val_loader)
    
    print("\n--- Training NeuMF (Separate Embeddings) ---")
    neumf_model = NeuralMatrixFactorization(num_users, num_items, mf_dim=8, mlp_dims=[32, 16, 8])
    trained_neumf = train_model(neumf_model, train_loader, val_loader)
    
    print("\n--- Training NeuMF (Shared Embeddings) ---")
    neumf_shared_model = NeuMF_Shared(num_users, num_items, embedding_dim=16, mlp_dims=[32, 16])
    trained_neumf_shared = train_model(neumf_shared_model, train_loader, val_loader)
    
    print("\n=== EVALUATING ALL MODELS ===")
    
    print("\n--- NCF Evaluation ---")
    ncf_metrics = evaluate_model(trained_ncf, test_loader, K=5)
    print(f"NCF Metrics: {ncf_metrics}")
    
    print("\n--- NeuMF (Separate) Evaluation ---")
    neumf_metrics = evaluate_model(trained_neumf, test_loader, K=5)
    print(f"NeuMF Metrics: {neumf_metrics}")
    
    print("\n--- NeuMF (Shared) Evaluation ---")
    neumf_shared_metrics = evaluate_model(trained_neumf_shared, test_loader, K=5)
    print(f"NeuMF Shared Metrics: {neumf_shared_metrics}")
    
    # Choose best model based on NDCG
    best_model = trained_neumf_shared  # Default choice
    best_metrics = neumf_shared_metrics
    best_name = "NeuMF_Shared"
    
    if neumf_metrics.get('NDCG@K', 0) > best_metrics.get('NDCG@K', 0):
        best_model = trained_neumf
        best_metrics = neumf_metrics
        best_name = "NeuMF_Separate"
    
    if ncf_metrics.get('NDCG@K', 0) > best_metrics.get('NDCG@K', 0):
        best_model = trained_ncf
        best_metrics = ncf_metrics
        best_name = "NCF"
    
    print(f"\n🏆 BEST MODEL: {best_name} with NDCG@5: {best_metrics.get('NDCG@K', 0):.4f}")
    
    print("\n=== COMPREHENSIVE ARCHITECTURE COMPARISON ===")
    arch_comparison_results, arch_comparison_obj = run_architecture_comparison(
        trained_ncf, trained_neumf, trained_neumf_shared, train_loader, test_loader
    )
    
    print("\n=== BEST MODEL SUMMARY ===")
    print(f"Winner: {best_name}")
    print(f"Precision@5: {best_metrics.get('Precision@K', 0):.4f}")
    print(f"Recall@5: {best_metrics.get('Recall@K', 0):.4f}")
    print(f"NDCG@5: {best_metrics.get('NDCG@K', 0):.4f}")
    
    # Skip traditional CF comparison due to memory constraints
    comparison_results = arch_comparison_results
    
    print("\n=== CREATING PRODUCTION SYSTEM ===")
    production_system = create_production_system(best_model, train_loader)
    
    # Example recommendations
    print("\n=== EXAMPLE RECOMMENDATIONS ===")
    # Get first user ID from training data
    sample_user = None
    for user, _, _ in train_loader:
        sample_user = user[0].item()
        break
    
    if sample_user is not None:
        recommendations = production_system.get_recommendations(sample_user, K=5)
        print(f"Recommendations for User {sample_user}:")
        for rec in recommendations:
            print(f"  - Product {rec['product_id']}: Score {rec['score']:.4f} (Rank {rec['rank']})")
    
    print("\n=== COLD START HANDLING ===")
    cold_start_handler = ColdStartHandler(best_model, train_loader.dataset)
    cold_user_recs = cold_start_handler.recommend_cold_user(K=5)
    print(f"Cold Start Recommendations: {cold_user_recs}")
    
    print("\n=== SAVING BEST MODEL ===")
    production_system.save_model_checkpoint("./saved_model")
    print(f"Best model ({best_name}) saved successfully!")
    
    return {
        'best_model': best_model,
        'best_name': best_name,
        'ncf_metrics': ncf_metrics,
        'neumf_metrics': neumf_metrics,
        'neumf_shared_metrics': neumf_shared_metrics,
        'comparison': comparison_results,
        'production_system': production_system
    }

if __name__ == "__main__":
    results = main()