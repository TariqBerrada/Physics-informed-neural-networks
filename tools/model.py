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
        data.requires_grad_()
        # a, b = self.batch_size, self.output_dim

        # output_grad =  torch.autograd.functional.jacobian(self, data).view((a*b, a*b))
        # output_dx = torch.stack((output_grad[:self.batch_size, :self.batch_size].diag(), output_grad[self.batch_size:, :self.batch_size].diag())).T
        ### print('dx requires grad ?', data.requires_grad)
        output= self.forward(data)
        u_dx = torch.autograd.grad(output[:, [0]], data, torch.ones((data.shape[0], 1)).to(self.device), retain_graph = True)[0]
        v_dx = torch.autograd.grad(output[:, [1]], data, torch.ones((data.shape[0], 1)).to(self.device), retain_graph = True)[0]

        output_dx = torch.stack((u_dx[:, 0], v_dx[:, 0])).T

        return output_dx

    def dt(self, data):
        data.requires_grad_()
        #### print('dt requires grad ?', data.requires_grad)
        # a, b = self.batch_size, self.output_dim

        # output_grad =  torch.autograd.functional.jacobian(self, data).view((a*b, a*b))
        # output_dt = torch.stack((output_grad[:self.batch_size, self.batch_size:].diag(), output_grad[self.batch_size:, self.batch_size:].diag())).T
        
        output= self.forward(data)
        u_dt = torch.autograd.grad(output[:, [0]], data, torch.ones((data.shape[0], 1)).to(self.device), retain_graph = True)[0]
        v_dt = torch.autograd.grad(output[:, [1]], data, torch.ones((data.shape[0], 1)).to(self.device), retain_graph = True)[0]

        output_dt = torch.stack((u_dt[:, 1], v_dt[:, 1])).T

        return output_dt

    def dx2(self, data):
        data.requires_grad_()
        output = self.forward(data)
        du = torch.autograd.grad(output[:, [0]], data, torch.ones((data.shape[0], 1)).to(self.device), create_graph = True)[0]
        dv = torch.autograd.grad(output[:, [1]], data, torch.ones((data.shape[0], 1)).to(self.device), create_graph  =True)[0]
        # du.requires_grad_()
        # dv.requires_grad_()
        #### print('d2u', du.requires_grad, data.requires_grad)

        d2u = torch.autograd.grad(du, data, torch.ones_like(data).to(self.device), create_graph = True)[0]
        d2v = torch.autograd.grad(dv, data, torch.ones_like(data).to(self.device), create_graph = True)[0]

        # output_dx2 = d2u[:, [0]]
        output_dx2 = torch.stack((d2u[:, 0], d2v[:, 0])).T

        return output_dx2
    
    def dt2(self, data):
        data.requires_grad_()
        output = self.forward(data)
        du = torch.autograd.grad(output[:, [0]], data, torch.ones((data.shape[0], 1)).to(self.device), create_graph = True)[0]
        dv = torch.autograd.grad(output[:, [1]], data, torch.ones((data.shape[0], 1)).to(self.device), create_graph  =True)[0]
        # du.requires_grad_()
        # dv.requires_grad_()
        ### print('d2u', du.requires_grad, data.requires_grad)

        d2u = torch.autograd.grad(du, data, torch.ones_like(data).to(self.device), create_graph = True)[0]
        d2v = torch.autograd.grad(dv, data, torch.ones_like(data).to(self.device), create_graph = True)[0]

        # output_dx2 = d2u[:, [0]]
        output_dt2 = torch.stack((d2u[:, 1], d2v[:, 1])).T

        return output_dt2

    # def dt2(self, data):
    #     data.requires_grad_()
    #     output = self.forward(data)
    #     du = torch.autograd.grad(output[:, [0]], data, torch.ones((data.shape[0], 1)).to(self.device), create_graph = True)[0]
    #     dv = torch.autograd.grad(output[:, [1]], data, torch.ones((data.shape[0], 1)).to(self.device), create_graph  =True)[0]
    #     # du.requires_grad_()
    #     # dv.requires_grad_()
    #     d2u = torch.autograd.grad(du, data, torch.ones_like(data).to(self.device), create_graph = True)[0]

    #     output_dt2 = d2u[:, [1]]
    #     return output_dt2

    # def dx2(self, data):
    #     a, b = self.batch_size, self.output_dim
        
    #     output_hess = torch.stack([torch.autograd.functional.hessian(Real(self, j), data).view(a*b, a*b) for j in range(a)])

    #     output_u_dx2 = output_hess[:, :a, :a].diagonal(dim1 = 1, dim2 = 2).diagonal()
    #     output_v_dx2 = output_hess[:, a:, a:].diagonal(dim1 = 1, dim2 = 2).diagonal()

    #     output_dx2 = torch.stack((output_u_dx2, output_v_dx2)).T
    
    #     return output_dx2

    # def dt2(self, data):
    #     a, b = self.batch_size, self.output_dim
        
    #     output_hess = torch.stack([torch.autograd.functional.hessian(Imag(self, j), data).view(a*b, a*b) for j in range(a)])

    #     output_u_dx2 = output_hess[:, :a, :a].diagonal(dim1 = 1, dim2 = 2).diagonal()
    #     output_v_dx2 = output_hess[:, a:, a:].diagonal(dim1 = 1, dim2 = 2).diagonal()

    #     output_dx2 = torch.stack((output_u_dx2, output_v_dx2)).T

    #     return output_dx2

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