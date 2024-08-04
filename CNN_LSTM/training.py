import torch
from imports import *
from constants import *

def train_model(model, train_loader, criterion, optimizer, num_epochs, early_stopping_patience):
    best_val_loss = float('inf')
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
        
        if train_loss < best_val_loss:
            best_val_loss = train_loss
            patience = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print("Early stopping")
                break
