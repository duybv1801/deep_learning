import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, val_loader, epochs=10, lr=0.005):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)  # L2 reg chống overfit
    
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 3

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, item, rating in train_loader:
            optimizer.zero_grad()
            pred = model(user, item)
            loss = criterion(pred, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user, item, rating in val_loader:
                pred = model(user, item)
                val_loss += criterion(pred, rating).item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    return model