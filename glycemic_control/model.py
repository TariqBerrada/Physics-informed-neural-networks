import torch

class GlycemicModel(torch.nn.Module):
    def __init__(self, n_layers = 5, hidden_dim = 100, activation = 'tanh', batch_size = 64):
        super(GlycemicModel, self).__init__()
        self.batch_size=  batch_size
        self.activation = torch.nn.Tanh() if activation == 'tanh' else torch.nn.ReLU()

           

        self.layers = [torch.nn.Linear(1, hidden_dim)]
        for _ in range(n_layers-2):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, 3))
        self.layers = torch.nn.ModuleList(self.layers)

        # Apply xavier initialization.
        for l in self.layers:
            torch.nn.init.xavier_uniform_(l.weight)

        self.p1 = 0.
        self.p2 = 0.025
        self.p3 = 0.013
        self.V1 = 12.
        self.n = 5/54
        self.Gb = 4.5
        self.Ib = 0.015

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device) 

    def forward(self, t):
        x = t
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        x = self.layers[-1](x)
        return x

    def G(self, t):
        return self.forward(t)[:, [0]]
    
    def I(self, t):
        return self.forward(t)[:, [1]]
    
    def X(self, t):
        return self.forward(t)[:, [2]]
    
    def P(self, t):
        return 0.5*torch.exp(-0.05*t)
    
    def G_dt(self, t):
        t.requires_grad_()
        G = self.G(t)
        G_t = torch.autograd.grad(G, t, torch.ones_like(t), create_graph = True, retain_graph = True)[0]
        
        return G_t
    
    def X_dt(self, t):
        t.requires_grad_()
        X = self.X(t)
        X_t = torch.autograd.grad(X, t, torch.ones_like(X), create_graph = True, retain_graph = True)[0]
        return X_t

    def I_dt(self, t):
        t.requires_grad_()
        I = self.I(t)
        I_t = torch.autograd.grad(I, t, torch.ones_like(I), create_graph = True, retain_graph = True)[0]
        return I_t
    
    def u(self, t):
        # To start off assume that G >= 6 mmol.L-1
        G = self.G(t)
        u_ = G*(0.41 - 0.0094*G)
        return u_

    def eq_1(self, t):
        return self.G_dt(t) + self.p1*self.G(t) + self.X(t)*(self.G(t)+self.Gb) - self.P(t)
    
    def eq_2(self, t):
        return self.X_dt(t) + self.p2*self.X(t) - self.p3*self.I(t)
    
    def eq_3(self, t):
        return self.I_dt(t) + self.n*(self.I(t) + self.Ib) - self.u(t)/self.V1

if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from glycemic_control.loss import l_0

    model = GlycemicModel()

    x = torch.zeros((32, 1))

    y = model(x)
    g = model.G(x)
    gdt = model.G_dt(x)
    print(y.shape, g.shape, gdt.shape)
    l = l_0(model)
    print(l)