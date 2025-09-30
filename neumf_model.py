import torch
import torch.nn as nn

class NeuralMatrixFactorization(nn.Module):
    """
    Neural Matrix Factorization (NeuMF) implementation
    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
    """
    def __init__(self, num_users, num_items, mf_dim=8, mlp_dims=[32, 16, 8], dropout_rate=0.2):
        super(NeuralMatrixFactorization, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        
        # GMF Embeddings (separate from MLP)
        self.user_mf_emb = nn.Embedding(num_users, mf_dim)
        self.item_mf_emb = nn.Embedding(num_items, mf_dim)
        
        # MLP Embeddings (separate from GMF)
        mlp_input_dim = mlp_dims[0] // 2  # Split equally for user and item
        self.user_mlp_emb = nn.Embedding(num_users, mlp_input_dim)
        self.item_mlp_emb = nn.Embedding(num_items, mlp_input_dim)
        
        # MLP Layers
        self.mlp_layers = nn.ModuleList()
        input_dim = mlp_dims[0]
        
        for dim in mlp_dims[1:]:
            self.mlp_layers.append(nn.Linear(input_dim, dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = dim
        
        # Final prediction layer
        final_input_dim = mf_dim + mlp_dims[-1]
        self.final_layer = nn.Linear(final_input_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with normal distribution"""
        nn.init.normal_(self.user_mf_emb.weight, std=0.01)
        nn.init.normal_(self.item_mf_emb.weight, std=0.01)
        nn.init.normal_(self.user_mlp_emb.weight, std=0.01)
        nn.init.normal_(self.item_mlp_emb.weight, std=0.01)
        
        # Initialize linear layers
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        nn.init.xavier_normal_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)
    
    def gmf_forward(self, user, item):
        """Generalized Matrix Factorization branch"""
        user_mf = self.user_mf_emb(user)  # [batch_size, mf_dim]
        item_mf = self.item_mf_emb(item)  # [batch_size, mf_dim]
        
        # Element-wise product (key operation in matrix factorization)
        gmf_output = user_mf * item_mf  # [batch_size, mf_dim]
        
        return gmf_output
    
    def mlp_forward(self, user, item):
        """Multi-Layer Perceptron branch"""
        user_mlp = self.user_mlp_emb(user)  # [batch_size, mlp_input_dim//2]
        item_mlp = self.item_mlp_emb(item)  # [batch_size, mlp_input_dim//2]
        
        # Concatenate user and item embeddings
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)  # [batch_size, mlp_input_dim]
        
        # Pass through MLP layers
        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        
        return mlp_output
    
    def forward(self, user, item):
        """
        Forward pass combining GMF and MLP
        
        Args:
            user: User indices [batch_size]
            item: Item indices [batch_size]
            
        Returns:
            Prediction scores [batch_size, 1]
        """
        # GMF branch: captures linear relationships
        gmf_out = self.gmf_forward(user, item)  # [batch_size, mf_dim]
        
        # MLP branch: captures non-linear relationships  
        mlp_out = self.mlp_forward(user, item)  # [batch_size, mlp_dims[-1]]
        
        # NeuMF: Fusion of GMF and MLP
        neumf_input = torch.cat([gmf_out, mlp_out], dim=-1)  # [batch_size, mf_dim + mlp_dims[-1]]
        
        # Final prediction
        output = self.final_layer(neumf_input)  # [batch_size, 1]
        
        return output.squeeze(-1)  # [batch_size]

# Alternative: NeuMF with shared embeddings (như mô tả của bạn)
class NeuMF_Shared(nn.Module):
    """
    NeuMF with shared embeddings between GMF and MLP branches
    """
    def __init__(self, num_users, num_items, embedding_dim=16, mlp_dims=[32, 16], dropout_rate=0.2):
        super(NeuMF_Shared, self).__init__()
        
        # Shared embeddings (như mô tả của bạn)
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        input_dim = embedding_dim * 2  # Concatenated user + item embeddings
        
        for dim in mlp_dims:
            self.mlp_layers.append(nn.Linear(input_dim, dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))
            input_dim = dim
        
        # Final layer combines GMF + MLP
        final_input_dim = embedding_dim + mlp_dims[-1]  # GMF output + MLP output
        self.final_layer = nn.Linear(final_input_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
        nn.init.xavier_normal_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)
    
    def forward(self, user, item):
        # Shared embeddings
        user_emb = self.user_emb(user)  # [batch_size, embedding_dim]
        item_emb = self.item_emb(item)  # [batch_size, embedding_dim]
        
        # GMF branch: element-wise product
        gmf_out = user_emb * item_emb  # [batch_size, embedding_dim]
        
        # MLP branch: concatenation + deep layers
        mlp_input = torch.cat([user_emb, item_emb], dim=-1)  # [batch_size, embedding_dim * 2]
        
        mlp_out = mlp_input
        for layer in self.mlp_layers:
            mlp_out = layer(mlp_out)  # [batch_size, mlp_dims[-1]]
        
        # Fusion
        neumf_input = torch.cat([gmf_out, mlp_out], dim=-1)
        output = self.final_layer(neumf_input)
        
        return output.squeeze(-1)
