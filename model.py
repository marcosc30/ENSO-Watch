import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# Model
class WeatherForecasterTCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size=3, dropout=0.2):
        super(WeatherForecasterTCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.kernel_size = kernel_size

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)

        # Pass through CNN layers
        x = self.cnn(x.permute(0, 2, 1))  # (batch_size, hidden_size, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, hidden_size)

        # Pass through LSTM layers
        _, (h_n, _) = self.lstm(x)  # (num_layers, batch_size, hidden_size)

        # Use the last hidden state from the LSTM
        out = self.fc(h_n[-1])  # (batch_size, output_size)

        return out    


    # Training function
    def train(model, dataloader, optimizer, criterion, device):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(dataloader)

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