import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.fc_input_layer = nn.Linear(input_size, hidden_dim) # linear transformation of input layer
        self.fc_hidden_layer = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)] # linear transformation of hidden layer
        self.fc_output_layer = nn.Linear(hidden_dim, output_size) # linear transformation of output layer
    
    def forward(self, x):
        x = self.fc_input_layer(x)
        for fc in self.fc_hidden_layer:
            x = fc(x)
        x = F.relu(x)
        x = self.fc_output_layer(x)
        x = F.softmax(x, dim=1)
        return x