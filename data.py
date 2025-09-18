import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

class RecDataset(Dataset):
    def __init__(self, users, items, ratings):
        self.users = torch.LongTensor(users)
        self.items = torch.LongTensor(items)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def load_and_preprocess_data(file_path='ecommerce_data.csv', implicit=True, neg_samples=4):
    # Load data
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['customer_id', 'product_id'])
    df['rating'] = (df['basket_count'] > 0).astype(float)  # Tất cả đều là 1 hiện tại
    
    # Map IDs
    user_map = {uid: i for i, uid in enumerate(df['customer_id'].unique())}
    item_map = {iid: i for i, iid in enumerate(df['product_id'].unique())}
    df['user_idx'] = df['customer_id'].map(user_map)
    df['item_idx'] = df['product_id'].map(item_map)
    
    num_users = len(user_map)
    num_items = len(item_map)
    
    # Chia train/val/test theo customer_id (70/15/15)
    unique_users = df['customer_id'].unique()
    train_users, temp_users = train_test_split(unique_users, test_size=0.3, random_state=42)
    val_users, test_users = train_test_split(temp_users, test_size=0.5, random_state=42)
    
    train_df = df[df['customer_id'].isin(train_users)]
    val_df = df[df['customer_id'].isin(val_users)]
    test_df = df[df['customer_id'].isin(test_users)]
    
    # Negative sampling cho implicit
    if implicit:
        def add_negatives(df, num_items, neg_samples):
            negatives = []
            for user in df['user_idx'].unique():
                user_pos = set(df[df['user_idx'] == user]['item_idx'])
                user_neg = np.random.choice([i for i in range(num_items) if i not in user_pos], 
                                          len(user_pos) * neg_samples, replace=False)
                negatives.extend([(user, item, 0.0) for item in user_neg])
            neg_df = pd.DataFrame(negatives, columns=['user_idx', 'item_idx', 'rating'])
            return pd.concat([df, neg_df])
        
        train_df = add_negatives(train_df, num_items, neg_samples)
        val_df = add_negatives(val_df, num_items, neg_samples)
        test_df = add_negatives(test_df, num_items, neg_samples)
    
    # Tạo loaders
    batch_size = 64
    train_loader = DataLoader(RecDataset(train_df['user_idx'].values, train_df['item_idx'].values, train_df['rating'].values), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RecDataset(val_df['user_idx'].values, val_df['item_idx'].values, val_df['rating'].values), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(RecDataset(test_df['user_idx'].values, test_df['item_idx'].values, test_df['rating'].values), batch_size=batch_size, shuffle=False)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    print(f"Train rating dist: {train_df['rating'].value_counts()}")
    print(f"Val rating dist: {val_df['rating'].value_counts()}")
    print(f"Test rating dist: {test_df['rating'].value_counts()}")
    val_ratings = val_loader.dataset.ratings.numpy()
    print(f"Val rating mean: {val_ratings.mean()}, unique: {np.unique(val_ratings)}")
    print(f"Val unique items: {len(val_df['item_idx'].unique())}")

    return train_loader, val_loader, test_loader, num_users, num_items