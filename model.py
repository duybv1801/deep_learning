import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=16, hidden_dims=[32, 16], dropout_rate=0.5):
        super(NCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        # Add batch normalization for better training
        self.bn = nn.BatchNorm1d(embedding_dim * 2)
        
        layers = []
        prev_dim = embedding_dim * 2
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            if i < len(hidden_dims) - 1:  # No activation after last hidden layer
                layers.append(nn.BatchNorm1d(dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        # Better initialization
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
    def forward(self, user, item):
        u_emb = self.user_emb(user)
        i_emb = self.item_emb(item)
        x = torch.cat([u_emb, i_emb], dim=-1)
        x = self.bn(x)  # Apply batch normalization
        return self.mlp(x).squeeze(-1)