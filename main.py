import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import WeatherForecasterCNNLSTM, WeatherForecasterCNNTransformer
from preprocess import preprocess_data

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1
    kernel_size = 3
    dropout = 0.2
    learning_rate = 0.001
    num_epochs = 10

    # Model
    model = WeatherForecasterCNNLSTM(input_size, hidden_size, num_layers, output_size, kernel_size, dropout).to(device)
    model = WeatherForecasterCNNTransformer(input_size, hidden_size, num_layers, output_size, kernel_size, dropout).to(device)

    # Dataset
    train_dataset, test_dataset,  = preprocess_data()
    # Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_cnn_lstm(model, train_data_loader, optimizer, criterion, device, num_epochs)
    train_cnn_transformer(model, train_data_loader, optimizer, criterion, device, num_epochs)

    # Test the model
    test_cnn_lstm = test_cnn_lstm(model, test_data_loader, criterion, device)
    test_cnn_transformer = test_cnn_transformer(model, test_data_loader, criterion, device)


def train_cnn_lstm(model, dataloader, optimizer, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")


def train_cnn_transformer(model, dataloader, optimizer, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")


def test_cnn_lstm(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)


def test_cnn_transformer(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(dataloader)


if __name__ == "__main__":
    main()