import torch

def loss_0(data, model):
    # data : (batch_size, 5)

    criterion = torch.nn.MSELoss(reduction = 'mean' )

    _input = torch.stack((data[:, 1], torch.zeros_like(data[:, 1]))).T

    h_init_pred = model(_input)
    h_init_tar = data[:, [2, 3]]
    MSE_0 = criterion(h_init_pred, h_init_tar)

    return MSE_0

def loss_b(data, model):
    criterion_b = torch.nn.MSELoss(reduction = 'mean')
    criterion_bx = torch.nn.MSELoss(reduction = 'mean')

    left_input = torch.stack((torch.ones_like(data[:, 1])*(-5), data[:, 1])).T
    right_input = torch.stack((torch.ones_like(data[:, 1])*(5), data[:, 1])).T

    left_output = model(left_input)
    right_output = model(right_input)

    # a, b = left_output.shape

    # left_output_grad =  torch.diagonal(torch.autograd.functional.jacobian(model, left_input).view((a*b, a*b))).view(a, b)
    # right_output_grad =  torch.diagonal(torch.autograd.functional.jacobian(model, right_input).view((a*b, a*b))).view(a, b)
    
    left_output_grad = model.dx(left_input)
    right_output_grad = model.dt(right_input)

    # print('shapes for training', left_output.shape, left_output_grad.shape)

    loss_mean = criterion_b(left_output, right_output)
    loss_grad = criterion_bx(left_output_grad, right_output_grad)

    return loss_mean + loss_grad


def loss_f(data, model, equation):
    criterion_f = torch.nn.MSELoss(reduction = 'mean')

    u_data = model(data[:, [2, 1]]) # x and then t.

    # a, b = u_data.shape
    # u_grad = torch.diagonal(torch.autograd.functional.jacobian(model))

    # u_grad_x = torch.autograd.grad(model(data[:, [2, 1]]), data[:, 2]) # not sure.
    # u_grad_t = torch.autograd.grad(model(data[:, [2, 1]]), data[:, 2])

    h = u_data# equation(data)
    # h_t = model.dt(data[:, [2, 1]])
    h_xx = model.dx2(data[:, [2, 1]])

    f_r = -h[:, 1] +.5*h_xx[:, 0] + h.pow(2).sum(axis = 1)*h[:, 0]
    f_im = h[:, 0] +.5*h_xx[:, 1] + h.pow(2).sum(axis = 1)*h[:, 1]

    f = f_r**2 + f_im**2

    MSE_f = criterion_f(f, torch.zeros_like(f))

    return MSE_f