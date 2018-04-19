import torch
import numpy as np


class Parameter(object):
    def __init__(self, init_tensor):
        self.data = init_tensor
        self.grad = torch.zeros(init_tensor.shape)

    def zero_grad(self):
        self.grad.zero_()

    def add_grad(self, grad_tensor):
        self.grad.add_(grad_tensor)


class Module(object):
    def __init__(self):
        super(Module, self).__init__()
        self.params = None
        self.activations = None
        self.outputs = None

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def get_params(self):
        return self.params

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        output = input.clamp(0, np.inf)
        self.outputs = output
        return output

    def backward(self, gradwrtoutput):
        g_output = (gradwrtoutput * self.outputs.sign()).sum(0)
        return g_output


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        self.outputs = input.tanh()
        return self.outputs

    def backward(self, gradwrtoutput):
        g_output = (gradwrtoutput * (1 - self.outputs**2)).sum(0)
        return g_output


class Linear(Module):
    def __init__(self, input_dim, output_dim, w_init=None, b_init=None):
        super(Linear, self).__init__()
        self.nb_units = output_dim
        self.input_dims = input_dim
        self.params = {
            'w': Parameter(torch.randn(output_dim, input_dim) if w_init is None else w_init),
            'b': Parameter(torch.zeros(output_dim) if b_init is None else b_init)
        }

    def forward(self, input):
        self.activations = input
        return (input @ self.params['w'].data.t()) + self.params['b'].data

    def backward(self, gradwrtoutput):
        self.params['b'].add_grad(gradwrtoutput.sum(0))
        self.params['w'].add_grad(self.activations.t()@gradwrtoutput)
        g_output = gradwrtoutput@self.params['w'].data
        return g_output.sum(0)

    def __repr__(self):
        return "Linear({})".format(str(self.nb_units))


class Sequential(Module):
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.layers = modules

    def __repr__(self):
        return "Sequential(\n  {})".format(",\n  ".join(self.layers))

    def forward(self , *input):
        out = input
        for l in self.layers:
            out = l(out)

        return out

    def backward(self, *gradwrtoutput):
        raise NotImplementedError
