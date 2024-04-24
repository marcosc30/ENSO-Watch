import torch
import torch.nn as nn

class WeatherForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size=3, dropout=0.2):
        super(WeatherForecaster, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.kernel_size = kernel_size
        
        # Temporal Convolutional Network (TCN) layers
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        
        # Pass through TCN layers
        x = self.tcn(x.permute(0, 2, 1))  # (batch_size, hidden_size, seq_len)
        
        # Pass through LSTM layers
        _, (h_n, _) = self.lstm(x)  # (num_layers, batch_size, hidden_size)
        
        # Use the last hidden state from the LSTM
        out = self.fc(h_n[-1])  # (batch_size, output_size)
        
        return out
