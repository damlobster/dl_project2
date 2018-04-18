import torch
import numpy as np


class Module(object):
    def __init__(self):
        super(Module, self).__init__()
        self.params = None
        self.gradients = None
        self.activations = None

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def get_params(self):
        return self.params

    def forward(self , *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        self.activations = input
        output = input.clamp(0, np.inf)
        return output

    def backward(self, gradwrtoutput):
        g_output = gradwrtoutput * self.activations.sign().sum(0)
        return g_output


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        self.activations = input
        output = input.tanh()
        return output

    def backward(self, gradwrtoutput):
        g_output = gradwrtoutput * (1 - self.activations.tanh()**2).sum(0)
        return g_output


class Linear(Module):
    def __init__(self, input_dim, output_dim, weights_init='normal'):
        super(Linear, self).__init__()
        self.nb_units = output_dim
        self.input_dims = input_dim
        self.params = (torch.FloatTensor(output_dim, input_dim).normal_(), torch.FloatTensor(output_dim).zero_())
        self.gradients = tuple(torch.FloatTensor(p.size()).zero_() for p in self.params)

    def forward(self, input):
        self.activations = input
        return (input @ self.params[0].t()) + self.params[1]

    def backward(self, gradwrtoutput):
        self.gradients[1] += gradwrtoutput
        self.gradients[0] += gradwrtoutput*self.activations
        g_output = gradwrtoutput*self.params[0]
        return g_output

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
