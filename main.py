import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import WeatherForecasterCNNLSTM, WeatherForecasterCNNTransformer, TemporalConvNet2D, ConvLSTM
from typing import Optional
# from preprocess import preprocess_data
from tqdm import tqdm
from visualize import visualize_results

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    input_size = 12
    hidden_size = 10240
    # hidden_size = 64
    num_layers = 2
    output_size = 12*4*4
    kernel_size = 3
    dropout = 0.2
    learning_rate = 0.002
    num_epochs = 10

    # Make each of the 4 models
    CNNLSTM = WeatherForecasterCNNLSTM(input_size, 64, num_layers, output_size, kernel_size, dropout).to(device)
    # CNNTransformer = WeatherForecasterCNNTransformer(input_size, 120, num_layers, output_size, kernel_size, dropout).to(device)
    TCN = TemporalConvNet2D(10, [10], kernel_size, dropout).to(device)
    # CLSTM = ConvLSTM(input_size, 120, kernel_size, num_layers).to(device)

    # Dataset
    # train_dataset, test_dataset = preprocess_data()
    train_dataset = torch.load('./data/train_dataset_norm_simple.pth')
    test_dataset = torch.load('./data/test_dataset_norm_simple.pth')

    # Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Loss and optimizers for each model
    criterion = nn.MSELoss()
    print(CNNLSTM.parameters())
    optimizer = Adam(CNNLSTM.parameters(), lr=learning_rate)

    # Lists to store training and test losses
    train_losses = [[], [], [], []]  # For each model
    test_losses = []
    model_names = ['CNNLSTM', 'CNNTransformer', 'TCN', 'CLSTM']

    #Train and test
    for epoch in range(num_epochs):
        # train_losses[0].append(train(CNNLSTM, train_data_loader, optimizer, criterion, device, epoch, num_epochs))
        # train_losses[1].append(train(CNNTransformer, train_data_loader, optimizer, criterion, device, epoch, num_epochs))
        train_losses[2].append(train(TCN, train_data_loader, optimizer, criterion, device, epoch, num_epochs))
        # train_losses[3].append(train(CLSTM, train_data_loader, optimizer, criterion, device, epoch, num_epochs))

    
    # test_losses.append(test(CNNLSTM, test_data_loader, criterion, device))
    # test_losses.append(test(CNNTransformer, test_data_loader, criterion, device))
    test_losses.append(test(TCN, test_data_loader, criterion, device))
    # test_losses.append(test(CLSTM, test_data_loader, criterion, device))

    # Visualize the results
    visualize_results(train_losses, test_losses, model_names, num_epochs)


def train(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{running_loss / len(dataloader):.4f}'})
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")
    return running_loss / len(dataloader)
        



def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)

if __name__ == "__main__":
    main()