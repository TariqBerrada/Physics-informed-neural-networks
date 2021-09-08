import torch

class NN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation = 'tanh'):
        super(NN, self).__init__()

        assert n_layers > 2, 'number of layers in model must be superior to 2, got %d!'%n_layers

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.activation = torch.nn.Tanh() if activation == 'tanh' else torch.nn.ReLU()

        self.layers = [torch.nn.Linear(input_dim, hidden_dim)]
        for _ in range(n_layers-2):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))

        # Apply xavier init initialization.
        for l in self.layers:
            torch.nn.init.xavier_uniform_(l.weight)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        x = self.layers[-1](x)
        return x

if __name__ =='__main__':
    model = NN(2, 2, 10, 5)

    example_input = torch.zeros(32, 2)
    out = model(example_input)
    
    print(out.shape)