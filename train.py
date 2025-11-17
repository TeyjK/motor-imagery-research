"""
Training script for motor imagery classification.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from models.classifier import Classifier
from utils.data_loader import get_data_loaders


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred_labels = torch.argmax(y_pred, dim=1)
        correct += (pred_labels == y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, loss_fn, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            
            total_loss += loss.item()
            pred_labels = torch.argmax(y_pred, dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def main(args):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data for subject {args.subject}...")
    train_loader, test_loader = get_data_loaders(
        args.subject, 
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    model = Classifier().to(device)
    print(f"Model architecture: {model.channel_list}")
    
    # Loss and optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    
    # Save model
    if args.save_model:
        os.makedirs('weights', exist_ok=True)
        model_path = f'weights/model_subject{args.subject:02d}.pt'
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train motor imagery classifier")
    parser.add_argument("--subject", type=int, required=True, help="Subject number (1-29)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_model", action="store_true", default=True, help="Save trained model")
    
    args = parser.parse_args()
    main(args)