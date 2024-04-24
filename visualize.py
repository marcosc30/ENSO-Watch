import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import Model

# Load the trained model
model = Model()
model.load_state_dict(torch.load('path_to_trained_model.pth'))
model.eval()

# Visualize the model architecture
def plot_model_architecture(model):
    # Generate a dummy input tensor
    dummy_input = torch.randn(1, 3, 32, 32)  # Modify the input shape to match model

    # Plot the model architecture
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title('Model Architecture')
    plt.imshow(plt.imread('path_to_model_architecture.png'))  # Modify the path to the model architecture image
    plt.show()

# Visualize the model's training history
def plot_training_history(history):
    # Plot training loss
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training accuracy
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Load the training history
history = torch.load('path_to_training_history.pth')

# Plot the model architecture
plot_model_architecture(model)

# Plot the training history
plot_training_history(history)
