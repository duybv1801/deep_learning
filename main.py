from data import load_and_preprocess_data
from model import NCF
from train import train_model
from evaluate import evaluate_model
from recommendation_system import create_production_system
from model_comparison import run_model_comparison
from cold_start import ColdStartHandler

def main():
    # Config
    file_path = 'ecommerce_data.csv'
    implicit = True
    
    print("=== LOADING AND PREPROCESSING DATA ===")
    train_loader, val_loader, test_loader, num_users, num_items = load_and_preprocess_data(
        file_path, implicit=implicit
    )
    
    print("\n=== TRAINING NCF MODEL ===")
    model = NCF(num_users, num_items)
    trained_model = train_model(model, train_loader, val_loader)
    
    print("\n=== EVALUATING MODEL ===")
    metrics = evaluate_model(trained_model, test_loader, K=5)
    print(f"\nFinal Metrics: {metrics}")
    
    print("\n=== MODEL COMPARISON ===")
    comparison_results, comparison_obj = run_model_comparison(
        trained_model, train_loader, test_loader
    )
    
    print("\n=== CREATING PRODUCTION SYSTEM ===")
    production_system = create_production_system(trained_model, train_loader)
    
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
    cold_start_handler = ColdStartHandler(trained_model, train_loader.dataset)
    cold_user_recs = cold_start_handler.recommend_cold_user(K=5)
    print(f"Cold Start Recommendations: {cold_user_recs}")
    
    print("\n=== SAVING MODEL ===")
    production_system.save_model_checkpoint("./saved_model")
    print("Model saved successfully!")
    
    return {
        'model': trained_model,
        'metrics': metrics,
        'comparison': comparison_results,
        'production_system': production_system
    }

if __name__ == "__main__":
    results = main()