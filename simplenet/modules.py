from torch import FloatTensor, LongTensor
import numpy as np


class Module(object):

    def __init__(self):
        super(Module, self).__init__()
        self.activations = None
        self.params = None

    def forward(self , *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def zero_grad(self):
        if self.params is not None:
            for p in self.params:
                p.zero_()

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def param(self):
        return self.params


class WithGrad(object):
    gradients = dict()

    def __init__(self):
        super(WithGrad, self).__init__()
        #self.gradients = None

    def init_grad(self):
        if WithGrad.gradients[self] is not None: return
        if not isinstance(self, Module):
            raise AttributeError(self.__class__.__name__+"() is not a Module")
        if self.params() is None:
            raise AttributeError(self.__class__.__name__+"() has no parameters")
        WithGrad.gradients[self] = tuple(FloatTensor(p.size()[1:]).zero_() for p in self.params())

    def zero_grad(self):
        for g in WithGrad.gradients[self]:
            g.zero_()

    @staticmethod
    def zero_all_grad():
        for module in WithGrad.gradients.values():
            module.zero_grad()

    def cum_grad(self, grad):
        for i in len(grad):
            WithGrad.gradients[self][i].add_(grad[i])

    def get_grad(self):
        return WithGrad.gradients[self]


class Functional(Module):

    def __init__(self):
        super(Functional, self).__init__()
        self.previous = None
        self.next = None

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs):
        previous = args[0]
        previous.next = self
        self.previous = previous
        out = self.forward(*args[1:])
        return out if type(out)!=tuple or len(out)>1 else out[0]

    def forward(self , *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError


class Sequential(Module):

    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.layers = modules

    def __repr__(self):
        return "Sequential(\n  {})".format(",\n  ".join(self.layers))

    def forward(self , *input):
        out = input
        for l in self.layers:
            out = l.forward(out)

        return out

    def backward(self, *gradwrtoutput):
        raise NotImplementedError



class ReLU(Functional):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, *inputs):
        self.activations = inputs
        outputs = tuple()
        for inp in inputs:
            outputs += (inp.clamp(0, np.inf),)
        return outputs

    def backward(self, *gradwrtoutput):
        g_outputs = tuple()
        for i, g_wrtoutput in enumerate(gradwrtoutput):
            g_outputs += g_wrtoutput * self.activations[i].sign().sum(0)
        return g_outputs


class Tanh(Functional):

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, *inputs):
        self.activations = inputs
        outputs = tuple()
        for inp in inputs:
            outputs += (inp.tanh(),)
        return outputs

    def backward(self, *gradwrtoutput):
        g_outputs = tuple()
        for i, g_wrtoutput in enumerate(gradwrtoutput):
            g_outputs += g_wrtoutput * (1 - self.activations[i].tanh()**2).sum(0)
        return g_outputs


class Linear(Functional, WithGrad):

    def __init__(self, nb_units, input_dims=None):
        super(Linear, self).__init__()
        self.nb_units = nb_units
        self.input_dims = input_dims
        if input_dims is not None:
            self.params = (FloatTensor(dim) for dim in input_dims)

    def forward(self, *inputs):
        if self.params is None:
            self.input_dims = (inp.shape[1:] for inp in inputs)
            self.params = tuple(FloatTensor(shape).normal_() for shape in self.input_dims)

        self.activations = tuple(inp*self.params[i] for i, inp in enumerate(inputs))
        return self.activations

    def backward(self, *gradwrtoutput):
        g_outputs = tuple()
        for i, g_wrtoutput in enumerate(gradwrtoutput):
            g_outputs += g_wrtoutput * (1 - self.activations[i].tanh()**2).sum(axis=0)
        return g_outputs

    def __repr__(self):
        return "Linear({})".format(str(self.nb_units))