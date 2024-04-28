import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# Model
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TemporalBlock2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock2D, self).__init__()

        # Convolutional layer with weight normalization
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Convolutional layer with weight normalization
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        
        # Sequential network
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout,
                                 self.conv2, self.relu, self.dropout)
        
        # Downsample layer
        self.downsample = nn.Conv2d(
            n_inputs, n_outputs, kernel_size=1, stride=stride) if n_inputs != n_outputs else None
        
        self.relu = nn.ReLU()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize weights with a normal distribution since we are using weight normalization
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        num_levels = len(num_channels)

        # Create a series of TemporalBlock2D layers
        for i in range(num_levels):
            # Calculate the dilation size
            dilation_size = 2 ** i
            # Create a TemporalBlock2D layer
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Append the TemporalBlock2D layer to the list of layers
            layers += [TemporalBlock2D(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=((kernel_size-1) * dilation_size, 0), dropout=dropout)]
            
        # Create a sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class WeatherForecasterCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size=3, dropout=0.2):
        super(WeatherForecasterCNNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.kernel_size = kernel_size

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size, padding='same'),
            nn.batch_norm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size, padding='same'),
            nn.batch_norm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (seq_len, batch_size, grid_data_length, grids_x, grids_y)
        # grids_x: number of grids in x direction
        # grids_y: number of grids in y direction

        # Separate all of the grids in the sequence
        convoluted_grids = []
        for grid in x:
            # Pass through CNN layers
            convoluted_grid = self.cnn(grid)
            convoluted_grids.append(convoluted_grid)
        lstm_out = self.lstm(torch.stack(convoluted_grids))
        out = self.fc(lstm_out)

        return out
    
'''
The input data is expected to have the shape (seq_len, batch_size, grid_data_length, grids_x, grids_y), where grids_x and grids_y represent 
the number of grids in the x and y directions, respectively.

The CNN part convolves over the spatial dimensions (e.g., grid data) and the output is permuted to have the shape (batch_size, seq_len, 
hidden_size) for the LSTM part.The CNN part processes each grid in the input sequence separately, producing a sequence of convoluted grids.

The LSTM part consists of a single LSTM layer with num_layers (default is 2) and hidden_size (default is 64). The LSTM processes the sequence
of hidden states from the CNN part, and the final hidden state h_n[-1] (from the last layer and last time step) is passed through a fully
connected layer to produce the output of size output_size (default is 1).
'''

class WeatherForecasterCNNTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size=3, dropout=0.2):
        super(WeatherForecasterCNNLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.kernel_size = kernel_size

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size, padding='same'),
            nn.batch_norm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size, padding='same'),
            nn.batch_norm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer layer
        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8),
            nn.TransformerEncoder(num_layers=num_layers, encoder_layer=self.transformer)
        )

        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (seq_len, batch_size, grid_data_length, grids_x, grids_y)
        # grids_x: number of grids in x direction
        # grids_y: number of grids in y direction

        # Separate all of the grids in the sequence
        convoluted_grids = []
        for grid in x:
            # Pass through CNN layers
            convoluted_grid = self.cnn(grid)
            convoluted_grids.append(convoluted_grid)
        transformer_out = self.transformer(torch.stack(convoluted_grids))
        out = self.fc(transformer_out)

        return out
    

'''
The Transformer part consists of a TransformerEncoderLayer and a TransformerEncoder with num_layers (default is 2) layers. 
The Transformer encoder processes the sequence of convoluted grids from the CNN part, attending to the spatial and temporal 
relationships within the input sequence. The output of the Transformer encoder is passed through a fully connected layer to
produce the final output.


In the WeatherForecasterTCN and WeatherForecasterCNNLSTM models, the timesteps of the LSTM correspond to the sequence length (seq_len) 
of the input data. The CNN convolves over the spatial dimensions (e.g., grid data) in each time step, extracting local features.

In the WeatherForecasterCNNTransformer model, the Transformer encoder attends to the entire sequence of convoluted grids from the CNN 
part, capturing both spatial and temporal relationships within the input sequence.
'''