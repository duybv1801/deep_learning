import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

class ModelComparison:
    """Compare different recommendation approaches"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name, model, description=""):
        """Add a model for comparison"""
        self.models[name] = {
            'model': model,
            'description': description,
            'results': {}
        }
    
    def compare_models(self, test_loader, metrics=['precision', 'recall', 'ndcg'], K=5):
        """Compare all models on test set"""
        from evaluate import evaluate_model
        
        results = {}
        for name, model_info in self.models.items():
            print(f"\n=== Evaluating {name} ===")
            print(f"Description: {model_info['description']}")
            
            # Evaluate model
            metrics_result = evaluate_model(model_info['model'], test_loader, K)
            results[name] = metrics_result
            
            # Store results
            self.models[name]['results'] = metrics_result
        
        # Print comparison table
        self._print_comparison_table(results, K)
        return results
    
    def _print_comparison_table(self, results, K):
        """Print formatted comparison table"""
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON - Top-{K} Recommendations")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'Precision@K':<12} {'Recall@K':<10} {'NDCG@K':<10} {'Coverage':<10}")
        print(f"{'-'*60}")
        
        for name, metrics in results.items():
            precision = metrics.get('Precision@K', 0)
            recall = metrics.get('Recall@K', 0)
            ndcg = metrics.get('NDCG@K', 0)
            coverage = metrics.get('Coverage', 0)
            
            print(f"{name:<15} {precision:<12.4f} {recall:<10.4f} {ndcg:<10.4f} {coverage:<10.4f}")

class TraditionalCF:
    """Traditional Collaborative Filtering for comparison"""
    
    def __init__(self, train_loader):
        self.user_item_matrix = self._build_matrix(train_loader)
        self.user_similarity = None
        self.item_similarity = None
        self._compute_similarities()
    
    def _build_matrix(self, train_loader):
        """Build user-item interaction matrix"""
        interactions = defaultdict(lambda: defaultdict(float))
        
        for user, item, rating in train_loader:
            for u, i, r in zip(user.numpy(), item.numpy(), rating.numpy()):
                interactions[u][i] = r
        
        return interactions
    
    def _compute_similarities(self):
        """Compute user-user and item-item similarities"""
        users = list(self.user_item_matrix.keys())
        items = set()
        for user_items in self.user_item_matrix.values():
            items.update(user_items.keys())
        items = list(items)
        
        # User-item matrix
        matrix = np.zeros((len(users), len(items)))
        user_idx = {u: i for i, u in enumerate(users)}
        item_idx = {i: idx for idx, i in enumerate(items)}
        
        for user, user_items in self.user_item_matrix.items():
            for item, rating in user_items.items():
                matrix[user_idx[user]][item_idx[item]] = rating
        
        # Compute similarities
        self.user_similarity = cosine_similarity(matrix)
        self.item_similarity = cosine_similarity(matrix.T)
        
        self.user_idx = user_idx
        self.item_idx = item_idx
        self.users = users
        self.items = items
    
    def predict(self, user, item):
        """Predict rating for user-item pair"""
        if user not in self.user_idx or item not in self.item_idx:
            return 0.5  # Default prediction
        
        u_idx = self.user_idx[user]
        i_idx = self.item_idx[item]
        
        # User-based CF
        user_sims = self.user_similarity[u_idx]
        numerator = 0
        denominator = 0
        
        for other_u_idx, sim in enumerate(user_sims):
            other_user = self.users[other_u_idx]
            if item in self.user_item_matrix[other_user] and sim > 0:
                numerator += sim * self.user_item_matrix[other_user][item]
                denominator += abs(sim)
        
        if denominator == 0:
            return 0.5
        
        return numerator / denominator
    
    def forward(self, user_tensor, item_tensor):
        """Interface compatible with neural models"""
        predictions = []
        for u, i in zip(user_tensor.numpy(), item_tensor.numpy()):
            pred = self.predict(u, i)
            predictions.append(pred)
        
        return torch.tensor(predictions)
    
    def __call__(self, user_tensor, item_tensor):
        """Make the model callable"""
        return self.forward(user_tensor, item_tensor)
    
    def eval(self):
        """Compatibility method"""
        pass

# Usage example for comparison
def run_model_comparison(best_model, train_loader, test_loader):
    """Run comprehensive model comparison"""
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Add best model (could be NCF, NeuMF, etc.)
    model_name = best_model.__class__.__name__
    comparison.add_model(
        name=model_name,
        model=best_model,
        description=f"Best performing model: {model_name}"
    )
    
    # Add traditional CF for baseline comparison
    traditional_cf = TraditionalCF(train_loader)
    comparison.add_model(
        name="Traditional_CF",
        model=traditional_cf,
        description="User-based Collaborative Filtering with cosine similarity"
    )
    
    # Compare models
    results = comparison.compare_models(test_loader, K=5)
    
    return results, comparison

def run_architecture_comparison(ncf_model, neumf_model, neumf_shared_model, train_loader, test_loader):
    """Compare different neural architectures"""
    
    comparison = ModelComparison()
    
    # Add all neural models
    comparison.add_model(
        name="NCF_MLP",
        model=ncf_model,
        description="MLP-only Neural Collaborative Filtering"
    )
    
    comparison.add_model(
        name="NeuMF_Separate",
        model=neumf_model,
        description="Neural Matrix Factorization with separate embeddings (GMF + MLP)"
    )
    
    comparison.add_model(
        name="NeuMF_Shared",
        model=neumf_shared_model,
        description="Neural Matrix Factorization with shared embeddings (GMF + MLP)"
    )
    
    # Add traditional baseline
    traditional_cf = TraditionalCF(train_loader)
    comparison.add_model(
        name="Traditional_CF",
        model=traditional_cf,
        description="Traditional Collaborative Filtering baseline"
    )
    
    # Compare all models
    results = comparison.compare_models(test_loader, K=5)
    
    return results, comparison
