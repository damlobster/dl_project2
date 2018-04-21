import torch
import numpy as np
#from .debug import log
import math

class Parameter(object):
    def __init__(self, init_tensor, name=None):
        self.data = init_tensor
        self.name = name
        self.grad = torch.zeros(init_tensor.shape)
        self.last_grad = torch.zeros(init_tensor.shape)

    def zero_grad(self):
        self.grad.zero_()

    def add_grad(self, grad_tensor):
        self.grad.add_(grad_tensor)

    def __repr__(self):
        return "{}: data {} grad {}".format(self.name, self.data)

class Module(object):
    def __init__(self):
        super(Module, self).__init__()
        self.params = None
        self.activations = None
        self.outputs = None

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def get_params(self):
        return self.params if self.params is None or isinstance(self.params, Parameter) else list(self.params.values())

    def zero_grads(self):
        if self.get_params() is not None:
            for p in self.get_params():
                p.zero_grad()

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
        g_output = (gradwrtoutput * self.outputs.sign())
        return g_output


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        self.outputs = input.tanh()
        return self.outputs

    def backward(self, gradwrtoutput):
        g_output = (gradwrtoutput * (1 - self.outputs**2))
        return g_output


class Linear(Module):
    def __init__(self, input_dim, output_dim, w_init=None, b_init=None):
        super(Linear, self).__init__()
        self.nb_units = output_dim
        self.input_dims = input_dim
        std = math.sqrt(2.0 / (output_dim + input_dim))
        ws = torch.zeros(output_dim, input_dim).normal_(0, std) if w_init is None else w_init
        bs = torch.zeros(output_dim).uniform_(-std, std) if b_init is None else b_init
        self.params = {
            'w': Parameter(ws, name=self.__repr__() + " w"),
            'b': Parameter(bs,  name=self.__repr__() + " b")
        }

    def forward(self, input):
        self.activations = input
        return (input @ self.params['w'].data.t()) + self.params['b'].data

    def backward(self, gradwrtoutput):
        self.params['b'].add_grad(gradwrtoutput.sum(0))
        self.params['w'].add_grad(self.activations.t()@gradwrtoutput)
        g_output = gradwrtoutput@self.params['w'].data
        return g_output

    def __repr__(self):
        return "Linear({}, {})".format(str(self.input_dims), str(self.nb_units))


class Sequential(Module):
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.layers = modules

    def __repr__(self):
        return "Sequential(\n  {})".format(",\n  ".join(self.layers))

    def forward(self , *input):
        out = input[0]
        for l in self.layers:
            out = l.forward(out)
        return out

    def backward(self, *gradwrtoutput):
        gradout = gradwrtoutput[0]
        for l in reversed(self.layers):
            gradout = l.backward(gradout)
        return gradout

    def zero_grads(self):
        for m in self.layers:
            m.zero_grads()

    def get_params(self):
        params = []
        for m in self.layers:
            p = m.get_params()
            if p is not None:
                params += p
        return params
