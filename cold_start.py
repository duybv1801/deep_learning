import torch
import numpy as np
from collections import defaultdict

class ColdStartHandler:
    """Handle cold start problem for new users and items"""
    
    def __init__(self, model, train_data):
        self.model = model
        self.train_data = train_data
        self.popular_items = self._get_popular_items()
        self.item_features = self._extract_item_features()
    
    def _get_popular_items(self, top_n=50):
        """Get most popular items based on interaction frequency"""
        item_counts = defaultdict(int)
        for _, item, rating in self.train_data:
            if rating > 0:
                item_counts[item.item()] += 1
        
        # Sort by popularity
        popular_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in popular_items[:top_n]]
    
    def _extract_item_features(self):
        """Extract item embeddings as features"""
        self.model.eval()
        item_features = {}
        with torch.no_grad():
            for item_id in range(self.model.item_emb.num_embeddings):
                item_features[item_id] = self.model.item_emb(torch.tensor([item_id]))
        return item_features
    
    def recommend_cold_user(self, K=5):
        """Recommend for completely new user (no history)"""
        # Strategy 1: Popular items
        return self.popular_items[:K]
    
    def recommend_cold_item(self, user_id, K=5):
        """Recommend when new item appears"""
        # Strategy 1: Use similar items based on embeddings
        # Strategy 2: Fall back to user's historical preferences
        
        # Get user's interaction history
        user_history = []
        for user, item, rating in self.train_data:
            if user.item() == user_id and rating > 0:
                user_history.append(item.item())
        
        if not user_history:
            return self.recommend_cold_user(K)
        
        # Find similar items to user's history
        recommendations = []
        for hist_item in user_history[-10]:  # Use last 10 interactions
            similar_items = self._find_similar_items(hist_item, K)
            recommendations.extend(similar_items)
        
        # Remove duplicates and return top-K
        unique_recs = list(dict.fromkeys(recommendations))
        return unique_recs[:K]
    
    def _find_similar_items(self, item_id, K=5):
        """Find similar items based on embedding similarity"""
        if item_id not in self.item_features:
            return []
        
        target_emb = self.item_features[item_id]
        similarities = {}
        
        for other_id, other_emb in self.item_features.items():
            if other_id != item_id:
                # Cosine similarity
                sim = torch.cosine_similarity(target_emb, other_emb, dim=0)
                similarities[other_id] = sim.item()
        
        # Sort by similarity
        similar_items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in similar_items[:K]]
