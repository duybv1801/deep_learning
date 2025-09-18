import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=16, hidden_dims=[16, 8], dropout_rate=0.2):
        super(NCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        layers = []
        prev_dim = embedding_dim * 2
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout_rate)])  # Chống overfit
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        # layers.append(nn.Sigmoid())  # Cho implicit probability
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
    def forward(self, user, item):
        u_emb = self.user_emb(user)
        i_emb = self.item_emb(item)
        x = torch.cat([u_emb, i_emb], dim=-1)
        return self.mlp(x).squeeze(-1)