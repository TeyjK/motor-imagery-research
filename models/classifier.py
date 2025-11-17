"""
Neural network models for EEG motor imagery classification.
"""

import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.
    
    Args:
        channel_list: List of integers specifying the number of neurons in each layer
        activation: Activation function class (default: ReLU)
        final_activation: Final layer activation function (default: Identity)
    """
    
    def __init__(self, channel_list=None, activation=None, final_activation=None):
        super().__init__()
        
        if channel_list is None:
            channel_list = [68, 2048, 1024, 512, 256, 128, 64, 2]
        
        if activation is None:
            activation = nn.ReLU
            
        if final_activation is None:
            final_activation = nn.Identity
        
        assert len(channel_list) > 1, "channel_list must have at least two elements"
        assert callable(activation), "activation must be callable"
        assert callable(final_activation), "final_activation must be callable"
        
        self.channel_list = channel_list
        self.activation = activation
        self.final_activation = final_activation
        
        # Build network layers
        self.linear_layers = nn.ModuleList()
        self.batchnorm_layers = nn.ModuleList()
        
        for i in range(len(channel_list) - 1):
            self.linear_layers.append(nn.Linear(channel_list[i], channel_list[i + 1]))
            self.batchnorm_layers.append(nn.BatchNorm1d(channel_list[i + 1]))
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        assert x.shape[-1] == self.channel_list[0], \
            f"Input must have shape [batch_size, {self.channel_list[0]}], got {x.shape}"
        
        x_shape = x.shape
        x = x.view(-1, self.channel_list[0])
        
        for i, (linear, batchnorm) in enumerate(zip(self.linear_layers, self.batchnorm_layers)):
            x = linear(x)
            x = batchnorm(x)
            
            # Apply final activation on last layer, otherwise use regular activation
            if i == len(self.linear_layers) - 1:
                x = self.final_activation()(x)
            else:
                x = self.activation()(x)
        
        x = x.view(x_shape[:-1] + (self.channel_list[-1],))
        return x


class Classifier(MultiLayerPerceptron):
    """
    Binary classifier for motor imagery classification.
    
    Extends MultiLayerPerceptron with softmax output for classification.
    Default architecture: [68, 2048, 1024, 512, 256, 128, 64, 2]
    """
    
    def __init__(self, channel_list=None, activation=None, final_activation=None):
        if channel_list is None:
            channel_list = [68, 2048, 1024, 512, 256, 128, 64, 2]
        
        if activation is None:
            activation = nn.ReLU
        
        if final_activation is None:
            final_activation = nn.Softmax
        
        super().__init__(channel_list, activation, final_activation)