import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from S4 import S4Block
from typing import Optional, Type, Union
from S4reqs import StandardEncoder


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTMCell, self).__init__()

        # Initialize the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Convolutional layers for the input, forget, output, and cell state
        self.conv = nn.Conv2d(
            in_channels=input_size + hidden_size,
            out_channels=4 * hidden_size,
            kernel_size=kernel_size,
            padding=self.padding,
        )

    # Define the forward pass
    def forward(self, input, prev_state):
        h_prev, c_prev = prev_state

        # Concatenate along the channel axis
        combined = torch.cat((input, h_prev), dim=1)
        combined_conv = self.conv(combined)

        # Split the combined convolutional tensor into separate input, forget, output, and cell state convolutions
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_size, dim=1)

        # Apply the activation functions
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Update the cell state and hidden state
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # Create a list of multiple ConvLSTM cells using nn.ModuleList because the number of layers is a parameter
        self.cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, input, prev_state=None):
        batch_size, seq_length, grid_data_length, height, width = input.size()

        # Initialize the hidden state and cell state for the first time step
        if prev_state is None:
            prev_state = [
                (
                    torch.zeros(batch_size, self.hidden_size, height, width),
                    torch.zeros(batch_size, self.hidden_size, height, width),
                )
                for _ in range(self.num_layers)
            ]

        # Initialize a list to store the hidden states and cell states for all layers
        next_state = []

        for t in range(seq_length):
            input_t = input[:, t, ...]

            layer_states = []
            # Iterate through each layer, passing the input and hidden states through the ConvLSTM cell for that layer
            for i, cell in enumerate(self.cells):
                h_prev, c_prev = prev_state[i]
                h_next, c_next = cell(input_t, (h_prev, c_prev))
                layer_states.append((h_next, c_next))
                input_t = h_next

        next_state.append(layer_states)
        out = torch.reshape(input_t, (batch_size, seq_length, grid_data_length, height, width))
        return out, next_state
    
class TemporalBlock2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock2D, self).__init__()

        # Convolutional layer with weight normalization
        self.conv1 = weight_norm(nn.Conv3d(n_inputs, n_outputs, (1, kernel_size, kernel_size),
                                           stride=(1, stride, stride), padding=padding, dilation=(1, dilation, dilation)))
        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Convolutional layer with weight normalization
        self.conv2 = weight_norm(nn.Conv3d(n_outputs, n_outputs, (1, kernel_size, kernel_size),
                                           stride=(1, stride, stride), padding=padding, dilation=(1, dilation, dilation)))
        
        # Sequential network
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout,
                                 self.conv2, self.relu, self.dropout)
        
        # Downsample layer
        self.downsample = nn.Sequential(
            nn.Conv3d(n_inputs, n_outputs, kernel_size=1, stride=(1, stride, stride)),
            nn.BatchNorm3d(n_outputs)
        ) if n_inputs != n_outputs else None
        
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
        # Pass the input through the network
        out = self.net(x)
        print(out.shape)
        # Add the input to the output (residual connection)
        res = x if self.downsample is None else self.downsample(x)
        print(res.shape)
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
                                       padding=(0, (kernel_size - 1) * dilation_size, (kernel_size - 1) * dilation_size), dropout=dropout)]
            
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
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_size*4*4, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Assume x has dimensions [batch_size, sequence_length, channels, height, width]
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        grid_data_length = x.shape[2]
        grids_x = x.shape[3]
        grids_y = x.shape[4]

        x = np.transpose(x, (1,0,2,3,4))

        # Separate all of the grids in the sequence
        convoluted_grids = []
        for grid in x:
            # Pass through CNN layers
            convoluted_grid = self.cnn(grid)
            convoluted_grids.append(convoluted_grid)

        # [batch_size, seq_length, grid_data_length * grids_x * grids_y]
        lstm_in = torch.reshape(torch.stack(convoluted_grids), (batch_size, seq_length, -1))

        # [batch_size, seq_length, hidden_size]
        lstm_out, _ = self.lstm(lstm_in)

        # [batch_size, hidden_size]
        lstm_out = lstm_out[:, -1, :]

        # [batch_size, output_size]
        out = self.fc(lstm_out)
        out = torch.reshape(out, (batch_size, 1, grid_data_length, grids_x, grids_y))

        # [batch_size, 1, grids_data_length, grids_x, grids_y]
        return out

class WeatherForecasterCNNTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size=3, dropout=0.2):
        super(WeatherForecasterCNNTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.kernel_size = kernel_size

        # Adjusted the input channel size to 12 if needed
        self.cnn = nn.Sequential(
            nn.Conv2d(12*10, hidden_size, kernel_size, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Resizing layer to match transformer's expected embedding dimension
        self.resize = nn.Linear(12*10*4*4, hidden_size)  

        # Transformer layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x : [batch_size, seq_length, grid_data_length, grids_x, grids_y]
        # Assume x has dimensions [batch_size, channels, height, width]
        batch_size, seq_length, grid_data_length, grids_x, grids_y = x.shape
        x = x.view((batch_size, -1, grids_x, grids_y))

        # Apply CNN layers
        x = self.cnn(x)  # Output shape [batch_size, hidden_size, new_height, new_width]

        # x : [batch_size, hidden_size * height * width]
        x = x.view(batch_size, -1)  # Flatten the output

        # Resize to correct transformer input dimension
        x = self.resize(x)  # Resize or project the flattened features to the embedding size

        # Prepare for transformer (add sequence length dimension)
        x = x.unsqueeze(1)  # Transformer expects [batch_size, seq_len, features]

        # Apply Transformer
        x = self.transformer(x)

        # Reshape and apply final fully connected layer
        x = x.view(batch_size, -1)  # Flatten the output of transformer
        x = self.fc(x)

        x = torch.reshape(x, (1, batch_size, grid_data_length, grids_x, grids_y))
        x = torch.permute(x, (1,0,2,3,4))

        return x

def _parse_pool_kernel(pool_kernel: Optional[Union[int, tuple[int]]]) -> int:
    if pool_kernel is None:
        return 1
    elif isinstance(pool_kernel, tuple):
        return pool_kernel[0]
    elif isinstance(pool_kernel, int):
        return pool_kernel
    else:
        raise TypeError(f"Unable to parse `pool_kernel`, got {pool_kernel}")


def _seq_length_schedule(
    n_blocks: int,
    l_max: int,
    pool_kernel: Optional[Union[int, tuple[int]]],
) -> list[tuple[int, int]]:
    ppk = _parse_pool_kernel(pool_kernel)

    schedule = list()
    for depth in range(n_blocks + 1):
        l_max_next = max(2, l_max // ppk)
        pool_ok = l_max_next > ppk
        schedule.append((l_max, pool_ok))
        l_max = l_max_next
    return schedule

class WeatherForecasterS4(nn.Module):
    """S4 Model.

    High-level implementation of the S4 model which:

        1. Encodes the input using the CNNs
        2. Applies ``1..n_blocks`` S4 blocks
        3. Decodes the output of step 2 using a linear layer

    Args:
        d_input (int): number of input features
        d_model (int): number of internal features
        d_output (int): number of features to return
        n_blocks (int): number of S4 blocks to construct
        n (int): dimensionality of the state representation
        l_max (int): length of input signal
        collapse (bool): if ``True`` average over time prior to
            decoding the result of the S4 block(s). (Useful for
            classification tasks.)
        p_dropout (float): probability of elements being set to zero
        activation (Type[nn.Module]): activation function to use after
            ``S4Layer()``.
        norm_type (str, optional): type of normalization to use.
            Options: ``batch``, ``layer``, ``None``.
        norm_strategy (str): position of normalization relative to ``S4Layer()``.
            Must be "pre" (before ``S4Layer()``), "post" (after ``S4Layer()``)
            or "both" (before and after ``S4Layer()``).
        pooling (nn.AvgPool1d, nn.MaxPool1d, optional): pooling method to use
            following each ``S4Block()``.

    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_blocks: int,
        n: int,
        l_max: int,
        collapse: bool = False,
        p_dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        norm_type: Optional[str] = "layer",
        norm_strategy: str = "post",
        pooling: Optional[Union[nn.AvgPool1d, nn.MaxPool1d]] = None,
        kernel_size=3
    ) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.n_blocks = n_blocks
        self.n = n
        self.l_max = l_max
        self.collapse = collapse
        self.p_dropout = p_dropout
        self.norm_type = norm_type
        self.norm_strategy = norm_strategy
        self.pooling = pooling

        *self.seq_len_schedule, (self.seq_len_out, _) = _seq_length_schedule(
            n_blocks=n_blocks,
            l_max=l_max,
            pool_kernel=None if self.pooling is None else self.pooling.kernel_size,
        )

        self.input_size = d_input
        self.hidden_size = d_model
        self.output_size = d_output
        self.kernel_size = kernel_size
        self.dropout = 0.2

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_size, self.hidden_size, self.kernel_size, padding='same'),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size, padding='same'),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.decoder = nn.Linear(self.d_model, self.d_output)
        self.blocks = nn.ModuleList(
            [
                S4Block(
                    d_model=d_model,
                    n=n,
                    l_max=seq_len,
                    p_dropout=p_dropout,
                    activation=activation,
                    norm_type=norm_type,
                    norm_strategy=norm_strategy,
                    pooling=pooling if pooling and pool_ok else None,
                )
                for (seq_len, pool_ok) in self.seq_len_schedule
            ]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, D_OUTPUT]`` if ``collapse``
                is ``True`` and ``[BATCH, SEQ_LEN // (POOL_KERNEL ** n_block), D_INPUT]``
                otherwise, where ``POOL_KERNEL`` is the kernel size of the ``pooling``
                layer. (Note that ``POOL_KERNEL=1`` if ``pooling`` is ``None``.)

        """
        y = self.encoder(u)
        #concatenate all of the feature vectors post CNN to make it acceptable to the S4 block.
        for block in self.blocks:
            y = block(y)
        return self.decoder(y.mean(dim=1) if self.collapse else y)