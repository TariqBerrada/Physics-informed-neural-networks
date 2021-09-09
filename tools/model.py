import torch, os

class NN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation = 'tanh', batch_size= 32):
        super(NN, self).__init__()

        assert n_layers > 2, 'number of layers in model must be superior to 2, got %d!'%n_layers

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

        self.activation = torch.nn.Tanh() if activation == 'tanh' else torch.nn.ReLU()

        self.layers = [torch.nn.Linear(input_dim, hidden_dim)]
        for _ in range(n_layers-2):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.layers = torch.nn.ModuleList(self.layers)

        # Apply xavier initialization.
        for l in self.layers:
            torch.nn.init.xavier_uniform_(l.weight)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        x = self.layers[-1](x)
        return x

    def load_weights(self, weights_dir):
        if os.path.isfile(weights_dir):
            state_dict = torch.load(weights_dir, map_location = 'cpu')
            state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict, strict = False)
            print('loaded state dict from : %s'%weights_dir)
        else:
            print('no file found at : %s'%weights_dir)


    def dx(self, data):
        a, b = self.batch_size, self.output_dim

        output_grad =  torch.autograd.functional.jacobian(self, data).view((a*b, a*b))
        output_dx = torch.stack((output_grad[:self.batch_size, :self.batch_size].diag(), output_grad[self.batch_size:, :self.batch_size].diag())).T
        return output_dx

    def dt(self, data):
        a, b = self.batch_size, self.output_dim

        output_grad =  torch.autograd.functional.jacobian(self, data).view((a*b, a*b))
        output_dt = torch.stack((output_grad[:self.batch_size, self.batch_size:].diag(), output_grad[self.batch_size:, self.batch_size:].diag())).T
        return output_dt

    def dx2(self, data):
        a, b = self.batch_size, self.output_dim
        
        output_hess = torch.stack([torch.autograd.functional.hessian(Real(self, j), data).view(a*b, a*b) for j in range(a)])

        output_u_dx2 = output_hess[:, :a, :a].diagonal(dim1 = 1, dim2 = 2).diagonal()
        output_v_dx2 = output_hess[:, a:, a:].diagonal(dim1 = 1, dim2 = 2).diagonal()

        output_dx2 = torch.stack((output_u_dx2, output_v_dx2)).T
    
        return output_dx2

    def dt2(self, data):
        a, b = self.batch_size, self.output_dim
        
        output_hess = torch.stack([torch.autograd.functional.hessian(Imag(self, j), data).view(a*b, a*b) for j in range(a)])

        output_u_dx2 = output_hess[:, :a, :a].diagonal(dim1 = 1, dim2 = 2).diagonal()
        output_v_dx2 = output_hess[:, a:, a:].diagonal(dim1 = 1, dim2 = 2).diagonal()

        output_dx2 = torch.stack((output_u_dx2, output_v_dx2)).T

        return output_dx2

def Real(model, j):
    def real_part(data):
        return model(data)[j, 0]
    return real_part

def Imag(model, j):
    def imag_part(data):
        return model(data)[j, 1]
    return imag_part

if __name__ =='__main__':
    model = NN(2, 2, 10, 5)

    example_input = torch.zeros(32, 2)
    out = model(example_input)

    print(out.shape)
    print(model.dx(example_input).shape)
    print(model.dt(example_input).shape)
    print(model.dx2(example_input).shape)
    print(model.dt2(example_input).shape)