import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, val_loader, epochs=15, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)  # Increased weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 5  # Increased patience

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_preds = 0
        total_samples = 0
        
        for user, item, rating in train_loader:
            optimizer.zero_grad()
            pred = model(user, item)
            loss = criterion(pred, rating)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Calculate training accuracy
            pred_probs = torch.sigmoid(pred)
            predictions = (pred_probs > 0.5).float()
            correct_preds += (predictions == rating).sum().item()
            total_samples += rating.size(0)
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for user, item, rating in val_loader:
                pred = model(user, item)
                val_loss += criterion(pred, rating).item()
                
                pred_probs = torch.sigmoid(pred)
                predictions = (pred_probs > 0.5).float()
                val_correct += (predictions == rating).sum().item()
                val_total += rating.size(0)
                
        val_loss /= len(val_loader)
        train_acc = correct_preds / total_samples
        val_acc = val_correct / val_total

        print(f'Epoch {epoch+1}, Train Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
    return model