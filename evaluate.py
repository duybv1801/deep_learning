import torch
import numpy as np

def evaluate_model(model, test_loader, K=5):
    model.eval()
    user_preds = {}
    user_trues = {}
    with torch.no_grad():
        for user, item, rating in test_loader:
            logits = model(user, item)
            preds = torch.sigmoid(logits)
            for u, i, p, r in zip(user.cpu().numpy(), item.cpu().numpy(), preds.cpu().numpy(), rating.cpu().numpy()):
                if u not in user_preds:
                    user_preds[u] = []
                    user_trues[u] = set()
                user_preds[u].append((i, p))
                if r > 0:  # Chỉ thêm item positive
                    user_trues[u].add(i)

    precision, recall = 0.0, 0.0
    n_users = len(user_preds)
    for u in user_preds:
        # Lấy top-K theo xác suất
        topk_preds = sorted(user_preds[u], key=lambda x: x[1], reverse=True)[:K]
        pred_items = {i for i, _ in topk_preds}

        # Lấy các item thực sự user đã tương tác (r > 0)
        true_items = user_trues[u]

        tp = len(pred_items & true_items)
        precision += tp / K if K > 0 else 0
        recall += tp / len(true_items) if len(true_items) > 0 else 0

    precision /= len(user_preds)
    recall /= len(user_preds)
    ctr = precision
    
    print(f'Test Precision@{K}: {precision:.4f}, Recall@{K}: {recall:.4f}, CTR: {ctr:.4f}')
    return {'Precision@K': precision, 'Recall@K': recall, 'CTR': ctr}