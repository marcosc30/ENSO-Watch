import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import WeatherForecasterCNNLSTM, WeatherForecasterCNNTransformer, TemporalConvNet2D, ConvLSTM
from typing import Optional
from tqdm import tqdm
from visualize import visualize_results

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    input_size = 1
    hidden_size_cnnlstm = 64
    hidden_size_cnntransformer = 120
    hidden_size_tcn = 64
    hidden_size_clstm = 64
    num_layers = 2
    output_size = 12 * 4 * 4
    kernel_size = 3
    dropout = 0.2
    learning_rate = 0.002
    num_epochs = 2
    num_features = 12



    # Dataset
    # train_dataset, test_dataset = preprocess_data()
    train_dataset = torch.load('./data/train_dataset_norm_simple.pth')
    test_dataset = torch.load('./data/test_dataset_norm_simple.pth')

    # Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Loss and optimizers for each model
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.HuberLoss()
    
    # Lists to store training and test losses
    train_losses = [] 
    test_losses = [0, 0, 0, 0]
    model_names = ['CNNLSTM', 'CNNTransformer', 'TCN', 'CLSTM']

    ## For each model: Instantiate, define optimizer, train, test loss

    # CNNLSTM = WeatherForecasterCNNLSTM(num_features, hidden_size_cnnlstm, num_layers, output_size, kernel_size, dropout).to(device)
    # cnnlstm_optimizer = Adam(CNNLSTM.parameters(), lr=learning_rate)
    # train_losses.append(train(CNNLSTM, train_data_loader, cnnlstm_optimizer, criterion, device, num_epochs))
    # test_losses[0] = test(CNNLSTM, test_data_loader, criterion, device)

    # CNNTransformer = WeatherForecasterCNNTransformer(num_features, hidden_size_cnntransformer, num_layers, output_size, kernel_size, dropout).to(device)
    # cnntransformer_optimizer = Adam(CNNTransformer.parameters(), lr=learning_rate)
    # train_losses.append(train(CNNTransformer, train_data_loader, cnntransformer_optimizer, criterion, device, num_epochs))
    # test_losses[1] = test(CNNTransformer, test_data_loader, criterion, device)

    # TCN = TemporalConvNet2D(input_size, num_features, kernel_size, dropout)
    # tcn_optimizer = Adam(TCN.parameters(), lr=learning_rate)
    # train_losses.append(train(TCN, train_data_loader, tcn_optimizer, criterion, device, num_epochs))
    # test_losses[2] = test(TCN, test_data_loader, criterion, device)

    # CLSTM = ConvLSTM(num_features, 120, kernel_size, num_layers).to(device)
    # clstm_optimizer = Adam(CLSTM.parameters(), lr=learning_rate)
    # train_losses.append(train(CLSTM, train_data_loader, clstm_optimizer, criterion, device, num_epochs))
    # test_losses[3] = test(CLSTM, test_data_loader, criterion, device)

    # Visualize the results
    visualize_results(train_losses, test_losses, model_names, num_epochs)


def train(model, dataloader, optimizer, criterion, device, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{running_loss / len(dataloader):.4f}'})
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")
        losses.append(running_loss/len(dataloader))
    return losses

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_correct, total_samples = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            num_correct += accuracy_within_tolerance(outputs, labels)
            total_samples += len(inputs)
    accuracy = (num_correct / total_samples).item()

    print('Test Accuracy: ', accuracy)
    return running_loss / len(dataloader)


def accuracy_within_tolerance(predictions, labels, tolerance=0.5, required_fraction=0.8):
    batch_size, _, num_features, height, width = predictions.shape
    
    differences = torch.abs(predictions - labels)
    within_tolerance = differences < tolerance
    
    num_elements = num_features * height * width
    fractions_within_tolerance = torch.sum(within_tolerance, axis=(1, 2, 3, 4)) / num_elements
    
    count = torch.sum(fractions_within_tolerance >= required_fraction)
    
    return count

if __name__ == "__main__":
    main()