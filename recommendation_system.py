import torch
import pickle
import numpy as np
from datetime import datetime
import json

class RecommendationSystem:
    """Production-ready recommendation system"""
    
    def __init__(self, model_path, metadata_path):
        self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.eval()
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.user_map = self.metadata['user_map']
        self.item_map = self.metadata['item_map']
        self.reverse_user_map = {v: k for k, v in self.user_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
    
    def get_recommendations(self, user_id, K=5, exclude_seen=True):
        """Get top-K recommendations for a user"""
        if user_id not in self.user_map:
            return self._handle_cold_user(K)
        
        user_idx = self.user_map[user_id]
        all_items = list(range(len(self.item_map)))
        
        # Get predictions for all items
        user_tensor = torch.tensor([user_idx] * len(all_items))
        item_tensor = torch.tensor(all_items)
        
        with torch.no_grad():
            logits = self.model(user_tensor, item_tensor)
            probs = torch.sigmoid(logits).numpy()
        
        # Create item-probability pairs
        item_probs = [(self.reverse_item_map[i], prob) for i, prob in enumerate(probs)]
        
        # Sort by probability
        item_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-K
        recommendations = []
        for item_id, prob in item_probs[:K]:
            recommendations.append({
                'product_id': item_id,
                'score': float(prob),
                'rank': len(recommendations) + 1
            })
        
        return recommendations
    
    def _handle_cold_user(self, K=5):
        """Handle new user with no history"""
        # Return popular items (this should be pre-computed)
        popular_items = ['popular_item_1', 'popular_item_2', 'popular_item_3', 'popular_item_4', 'popular_item_5']
        return [{'product_id': item, 'score': 0.5, 'rank': i+1} for i, item in enumerate(popular_items[:K])]
    
    def batch_recommend(self, user_ids, K=5):
        """Get recommendations for multiple users"""
        results = {}
        for user_id in user_ids:
            results[user_id] = self.get_recommendations(user_id, K)
        return results
    
    def save_model_checkpoint(self, save_path):
        """Save model and metadata"""
        torch.save(self.model, f"{save_path}/model.pth")
        
        with open(f"{save_path}/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save model info
        model_info = {
            'created_at': datetime.now().isoformat(),
            'num_users': len(self.user_map),
            'num_items': len(self.item_map),
            'model_type': 'NCF',
            'version': '1.0'
        }
        
        with open(f"{save_path}/model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)

def create_production_system(model, train_loader):
    """Create production-ready system from trained model"""
    # Extract mappings from training data
    user_set = set()
    item_set = set()
    
    for user, item, _ in train_loader:
        user_set.update(user.numpy())
        item_set.update(item.numpy())
    
    user_map = {uid: i for i, uid in enumerate(sorted(user_set))}
    item_map = {iid: i for i, iid in enumerate(sorted(item_set))}
    
    metadata = {
        'user_map': user_map,
        'item_map': item_map,
        'created_at': datetime.now().isoformat()
    }
    
    # Save everything
    torch.save(model, 'model.pth')
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    return RecommendationSystem('model.pth', 'metadata.pkl')
