"""
Evaluation script for trained motor imagery classifier.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from models.classifier import Classifier
from utils.data_loader import get_data_loaders


def evaluate_model(model, test_loader, device):
    """
    Evaluate model and collect predictions.
    
    Returns:
        Tuple of (all_predictions, all_labels, accuracy)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            pred_labels = torch.argmax(y_pred, dim=1)
            
            all_predictions.extend(pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    return np.array(all_predictions), np.array(all_labels), accuracy


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Left (0)', 'Right (1)']
    )
    disp.plot(cmap='Blues')
    plt.title('Motor Imagery Classification - Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main(args):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading test data for subject {args.subject}...")
    _, test_loader = get_data_loaders(
        args.subject,
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    
    # Load model
    model = Classifier().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, labels, accuracy = evaluate_model(model, test_loader, device)
    
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'='*50}\n")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(
        labels, 
        predictions,
        target_names=['Left Hand', 'Right Hand'],
        digits=4
    ))
    
    # Confusion matrix
    if args.plot_confusion:
        plot_confusion_matrix(labels, predictions, args.confusion_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate motor imagery classifier")
    parser.add_argument("--subject", type=int, required=True, help="Subject number (1-29)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--plot_confusion", action="store_true", help="Plot confusion matrix")
    parser.add_argument("--confusion_path", type=str, default="confusion_matrix.png", 
                        help="Path to save confusion matrix")
    
    args = parser.parse_args()
    main(args)