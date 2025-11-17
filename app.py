"""
Interactive training dashboard for motor imagery classification.
Run with: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix
import numpy as np

from models.classifier import Classifier
from utils.data_loader import get_data_loaders


def train_with_logging(model, train_loader, test_loader, epochs, lr, device):
    """Train model and return metrics for visualization."""
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (torch.argmax(y_pred, dim=1) == y).sum().item()
            train_total += y.size(0)
        
        # Testing
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                
                test_loss += loss.item()
                test_correct += (torch.argmax(y_pred, dim=1) == y).sum().item()
                test_total += y.size(0)
        
        # Log metrics
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['test_loss'].append(test_loss / len(test_loader))
        history['test_acc'].append(test_correct / test_total)
        
        # Update UI
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} | "
                        f"Test Acc: {history['test_acc'][-1]:.4f}")
    
    return history, model


def plot_training_curves(history):
    """Plot loss and accuracy curves."""
    df = pd.DataFrame(history)
    
    # Loss plot
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=df['epoch'], y=df['train_loss'],
                                  mode='lines+markers', name='Train Loss'))
    fig_loss.add_trace(go.Scatter(x=df['epoch'], y=df['test_loss'],
                                  mode='lines+markers', name='Test Loss'))
    fig_loss.update_layout(title='Loss Curves', xaxis_title='Epoch', yaxis_title='Loss')
    
    # Accuracy plot
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=df['epoch'], y=df['train_acc'],
                                 mode='lines+markers', name='Train Accuracy'))
    fig_acc.add_trace(go.Scatter(x=df['epoch'], y=df['test_acc'],
                                 mode='lines+markers', name='Test Accuracy'))
    fig_acc.update_layout(title='Accuracy Curves', xaxis_title='Epoch', yaxis_title='Accuracy')
    
    return fig_loss, fig_acc


def plot_confusion_matrix(model, test_loader, device):
    """Generate confusion matrix from test data."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            all_preds.extend(torch.argmax(y_pred, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Left (0)', 'Right (1)'],
                    y=['Left (0)', 'Right (1)'],
                    text_auto=True,
                    color_continuous_scale='Blues')
    fig.update_layout(title='Confusion Matrix')
    
    return fig


def main():
    st.set_page_config(page_title="Motor Imagery Classifier", layout="wide")
    
    st.title("Motor Imagery Classification Dashboard")
    st.markdown("Train and visualize a neural network for EEG-based motor imagery classification")
    
    # Sidebar controls
    st.sidebar.header("Training Configuration")
    subject = st.sidebar.selectbox("Subject Number", range(1, 5), index=0)
    epochs = st.sidebar.slider("Training Epochs", 1, 50, 10)
    learning_rate = st.sidebar.select_slider("Learning Rate", 
                                             options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                                             value=1e-4,
                                             format_func=lambda x: f"{x:.0e}")
    batch_size = st.sidebar.selectbox("Batch Size", [256, 512, 1024, 2048], index=2)
    
    # Model architecture display
    st.sidebar.header("Model Architecture")
    st.sidebar.code("Input (68)\n↓\n2048 → ReLU\n↓\n1024 → ReLU\n↓\n512 → ReLU\n↓\n256 → ReLU\n↓\n128 → ReLU\n↓\n64 → ReLU\n↓\nOutput (2)")
    
    # Main training section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Training Progress")
        
    with col2:
        st.header("Model Info")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        st.metric("Device", str(device))
        st.metric("Parameters", "~10M")
    
    if st.button("Start Training", type="primary"):
        # Load data
        with st.spinner(f"Loading data for subject {subject}..."):
            train_loader, test_loader = get_data_loaders(subject, batch_size=batch_size)
        
        st.success(f"Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")
        
        # Initialize model
        model = Classifier().to(device)
        
        # Train
        st.subheader("Training in progress...")
        history, trained_model = train_with_logging(model, train_loader, test_loader, 
                                                     epochs, learning_rate, device)
        
        # Display results
        st.success(f"Training complete! Final test accuracy: {history['test_acc'][-1]:.4f}")
        
        # Plot curves
        st.subheader("Training Metrics")
        fig_loss, fig_acc = plot_training_curves(history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_loss, use_container_width=True)
        with col2:
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig_cm = plot_confusion_matrix(trained_model, test_loader, device)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Download model
        st.subheader("Save Model")
        if st.button("Download Trained Model"):
            torch.save(trained_model.state_dict(), f"model_subject{subject}.pt")
            st.success(f"Model saved as model_subject{subject}.pt")


if __name__ == "__main__":
    main()
