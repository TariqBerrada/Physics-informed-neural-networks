import torch, os

import matplotlib.pyplot as plt

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

    def forward(self, x, t):
        d = torch.stack([x, t], dim = 1)
        for l in self.layers[:-1]:
            d = self.activation(l(d))
        d = self.layers[-1](d)
        return d

    def load_weights(self, weights_dir):
        if os.path.isfile(weights_dir):
            state_dict = torch.load(weights_dir, map_location = 'cpu')
            state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict, strict = False)
            print('loaded state dict from : %s'%weights_dir)
        else:
            print('no file found at : %s'%weights_dir)


    def dx(self, data):
        data.requires_grad_()
        x = data[:, 0]
        t = data[:, 1]

        u = self.real(x, t)
        v = self.imag(x, t)
    
        u_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph = True, retain_graph = True)[0]
        v_dx = torch.autograd.grad(v, x, torch.ones_like(v), create_graph = True, retain_graph = True)[0]
        
        output_dx = torch.stack((u_dx, v_dx), dim = 1)


        return output_dx

    def dt(self, data):
        data.requires_grad_()

        x = data[:, 0]
        t = data[:, 1]

        u = self.real(x, t)
        v = self.imag(x, t)

        u_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph = True, retain_graph = True)[0]
        v_dt = torch.autograd.grad(v, t, torch.ones_like(v), create_graph = True, retain_graph = True)[0]

        output_dt = torch.stack((u_dt, v_dt), dim = 1)

        return output_dt

    def dx2(self, data):
        data.requires_grad_()
        
        x = data[:, 0]
        t = data[:, 1]

        u = self.real(x, t)
        v = self.imag(x, t)

        u_dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph = True, retain_graph = True)[0]
        v_dx = torch.autograd.grad(v, x, torch.ones_like(v), create_graph = True, retain_graph = True)[0]
        
        u_dx2 = torch.autograd.grad(u_dx, x, torch.ones_like(u_dx), create_graph = True, retain_graph = True)[0]
        v_dx2 = torch.autograd.grad(v_dx, x, torch.ones_like(v_dx), create_graph = True, retain_graph = True)[0]

        output_dx2 = torch.stack((u_dx2, v_dx2), dim = 1)

        return output_dx2
    
    def dt2(self, data):
        data.requires_grad_()
        
        x = data[:, 0]
        t = data[:, 1]

        u = self.real(x, t)
        v = self.imag(x, t)

        u_dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph = True, retain_graph = True)[0]
        v_dt = torch.autograd.grad(v, t, torch.ones_like(v), create_graph = True, retain_graph = True)[0]

        u_dt2 = torch.autograd.grad(u_dt, t, torch.ones_like(u_dt), create_graph = True, retain_graph = True)[0]
        v_dt2 = torch.autograd.grad(v_dt, t, torch.ones_like(v_dt), create_graph = True, retain_graph = True)[0]

        output_dt2 = torch.stack((u_dt2, v_dt2), dim = 1)

        return output_dt2
    
    def real(self, x, t):
        return self.forward(x, t)[:, 0]

    def imag(self, x, t):
        return self.forward(x, t)[:, 1]


def Real(model, j):
    def real_part(data):
        return model(data)[j, 0]
    return real_part

def Imag(model, j):
    def imag_part(data):
        return model(data)[j, 1]
    return imag_part

if __name__ =='__main__':
    device = 'cuda'
    model = NN(2, 2, 50, 5).to(device)
    model.load_weights('./weights/basic.pth.tar')

    example_input = torch.ones(32, 2, requires_grad = True, device = device)
    example_input[:, 1] = torch.linspace(0, 1, 32)
    print('e', example_input)
    out = model(example_input)

    print(out)
    print('----')
    print(model.dx(example_input))
    print('----')
    print(model.dt(example_input))
    print('----')
    print(model.dx2(example_input))
    print('----')
    print(model.dt2(example_input))

    out = out.detach().cpu().numpy()
    plt.figure()
    plt.plot(out[:, 0])
    plt.plot(out[:, 1])
    plt.show()