import torch.nn as nn

class ReversiValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(129, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh()
        )
    
import numpy as np

class FastNumpyNet:
    def __init__(self, pytorch_model):
        # Extract weights and biases (Skipping the odd indices which are ReLU/Tanh)
        self.w1 = pytorch_model.layers[0].weight.detach().numpy()
        self.b1 = pytorch_model.layers[0].bias.detach().numpy()
        
        self.w2 = pytorch_model.layers[2].weight.detach().numpy()
        self.b2 = pytorch_model.layers[2].bias.detach().numpy()
        
        self.w3 = pytorch_model.layers[4].weight.detach().numpy()
        self.b3 = pytorch_model.layers[4].bias.detach().numpy()
        
        self.w4 = pytorch_model.layers[6].weight.detach().numpy()
        self.b4 = pytorch_model.layers[6].bias.detach().numpy()
        
        self.w5 = pytorch_model.layers[8].weight.detach().numpy()
        self.b5 = pytorch_model.layers[8].bias.detach().numpy()

    def predict(self, x):
        # Perform the neural network layers manually
        x = np.maximum(0, np.dot(x, self.w1.T) + self.b1) # ReLU
        x = np.maximum(0, np.dot(x, self.w2.T) + self.b2) # ReLU
        x = np.maximum(0, np.dot(x, self.w3.T) + self.b3) # ReLU
        x = np.maximum(0, np.dot(x, self.w4.T) + self.b4) # ReLU
        x = np.tanh(np.dot(x, self.w5.T) + self.b5)       # Tanh
        return x[0][0]