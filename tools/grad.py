import torch

def dx(model, data):
    output = model(data) 
    a, b = output.shape

    output_grad =  torch.diagonal(torch.autograd.functional.jacobian(model, output).view((a*b, a*b))).view(a, b)
    return output_grad