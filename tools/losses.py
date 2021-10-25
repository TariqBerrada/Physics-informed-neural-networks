import torch

criterion = torch.nn.MSELoss(reduction = 'mean' )

criterion_b = torch.nn.MSELoss(reduction = 'mean')
criterion_bx = torch.nn.MSELoss(reduction = 'mean')

criterion_f = torch.nn.MSELoss(reduction = 'mean')


def loss_0(data, model):

    _input = torch.stack((data[:, 1], torch.zeros_like(data[:, 1]))).T

    h_init_pred = model(x = _input[:, 0], t = _input[:, 1])
    h_init_tar = data[:, [2, 3]]
    MSE_0 = criterion(h_init_pred, h_init_tar)

    return MSE_0

def loss_b(data, model):

    left_input = torch.stack((torch.ones_like(data[:, 1])*(-5), data[:, 1])).T.requires_grad_()
    right_input = torch.stack((torch.ones_like(data[:, 1])*(5), data[:, 1])).T.requires_grad_()

    left_output = model(x = left_input[:, 0], t = left_input[:, 1])
    right_output = model(x = right_input[:, 0], t = right_input[:, 1])

    # a, b = left_output.shape

    # left_output_grad =  torch.diagonal(torch.autograd.functional.jacobian(model, left_input).view((a*b, a*b))).view(a, b)
    # right_output_grad =  torch.diagonal(torch.autograd.functional.jacobian(model, right_input).view((a*b, a*b))).view(a, b)
    
    left_output_grad = model.dx(left_input)
    right_output_grad = model.dt(right_input)

    loss_mean = criterion_b(left_output, right_output)
    loss_grad = criterion_bx(left_output_grad, right_output_grad)

    return loss_mean + loss_grad


def loss_f(data, model, equation):
    x = data[:, 3]
    t = data[:, 2]

    # u_data = model(data[:, [3, 2]]) # x and then t.

    u_data = model(x, t)

    # a, b = u_data.shape
    # u_grad = torch.diagonal(torch.autograd.functional.jacobian(model))

    # u_grad_x = torch.autograd.grad(model(data[:, [2, 1]]), data[:, 2]) # not sure.
    # u_grad_t = torch.autograd.grad(model(data[:, [2, 1]]), data[:, 2])

    h = u_data # equation(data)
    h_t = model.dt(data[:, [3, 2]])
    h_xx = model.dx2(data[:, [3, 2]])

    f_r = -h_t[:, 1] +.5*h_xx[:, 0] + h.pow(2).sum(axis = 1)*h[:, 0]
    f_im = h_t[:, 0] +.5*h_xx[:, 1] + h.pow(2).sum(axis = 1)*h[:, 1]

    f = f_r**2 + f_im**2

    MSE_f = f.mean()

    return MSE_f