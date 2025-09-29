import torch
import numpy as np

def evaluate_model(model, test_loader, K=5):
    model.eval()
    user_preds = {}
    user_trues = {}
    
    # Debug: Track prediction statistics
    all_preds = []
    all_ratings = []
    
    with torch.no_grad():
        for user, item, rating in test_loader:
            logits = model(user, item)
            preds = torch.sigmoid(logits)
            
            # Collect for debugging
            all_preds.extend(preds.cpu().numpy())
            all_ratings.extend(rating.cpu().numpy())
            
            for u, i, p, r in zip(user.cpu().numpy(), item.cpu().numpy(), preds.cpu().numpy(), rating.cpu().numpy()):
                if u not in user_preds:
                    user_preds[u] = []
                    user_trues[u] = set()
                user_preds[u].append((i, p))
                if r > 0:  # Chỉ thêm item positive
                    user_trues[u].add(i)

    # Debug output
    all_preds = np.array(all_preds)
    all_ratings = np.array(all_ratings)
    
    # Find optimal threshold
    thresholds = np.arange(0.1, 0.9, 0.1)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        pred_binary = (all_preds > thresh).astype(int)
        tp = ((pred_binary == 1) & (all_ratings == 1)).sum()
        fp = ((pred_binary == 1) & (all_ratings == 0)).sum()
        fn = ((pred_binary == 0) & (all_ratings == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    precision, recall = 0.0, 0.0
    n_users = len(user_preds)
    valid_users = 0  # Users with at least one positive item
    
    for u in user_preds:
        # Lấy top-K theo xác suất
        topk_preds = sorted(user_preds[u], key=lambda x: x[1], reverse=True)[:K]
        pred_items = {i for i, _ in topk_preds}

        # Lấy các item thực sự user đã tương tác (r > 0)
        true_items = user_trues[u]
        
        if len(true_items) > 0:  # Only count users with positive items
            valid_users += 1
            tp = len(pred_items & true_items)
            precision += tp / K if K > 0 else 0
            recall += tp / len(true_items)

    if valid_users > 0:
        precision /= valid_users
        recall /= valid_users
    else:
        precision = recall = 0.0
        
    ctr = precision
    
    # Calculate additional metrics
    ndcg = calculate_ndcg(user_preds, user_trues, K)
    coverage = len(set([item for user_items in user_preds.values() for item, _ in user_items])) / len(set([item for user_items in user_trues.values() for item in user_items]))
    
    print(f"Debug - Valid users (with positive items): {valid_users}/{n_users}")
    print(f"Debug - Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    print(f"Debug - Positive predictions (0.5): {(all_preds > 0.5).sum()}/{len(all_preds)} = {(all_preds > 0.5).mean():.4f}")
    print(f"Debug - Positive predictions (optimal): {(all_preds > best_threshold).sum()}/{len(all_preds)} = {(all_preds > best_threshold).mean():.4f}")
    print(f'Test Precision@{K}: {precision:.4f}, Recall@{K}: {recall:.4f}, NDCG@{K}: {ndcg:.4f}, Coverage: {coverage:.4f}')
    return {'Precision@K': precision, 'Recall@K': recall, 'NDCG@K': ndcg, 'Coverage': coverage}

def calculate_ndcg(user_preds, user_trues, K):
    """Calculate Normalized Discounted Cumulative Gain"""
    total_ndcg = 0
    valid_users = 0
    
    for u in user_preds:
        if len(user_trues[u]) == 0:
            continue
            
        # Get top-K predictions
        topk_preds = sorted(user_preds[u], key=lambda x: x[1], reverse=True)[:K]
        
        # Calculate DCG
        dcg = 0
        for i, (item, _) in enumerate(topk_preds):
            if item in user_trues[u]:
                dcg += 1 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(user_trues[u]), K)))
        
        # NDCG
        if idcg > 0:
            total_ndcg += dcg / idcg
            valid_users += 1
    
    return total_ndcg / valid_users if valid_users > 0 else 0