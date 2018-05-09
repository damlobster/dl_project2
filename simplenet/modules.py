import torch
import math

class Parameter(object):
    r"""Hold the data and gradient tensor for a parameter of a module"""
    def __init__(self, init_tensor, name=None):
        self.data = init_tensor
        self.name = name
        self.grad = torch.zeros(init_tensor.shape)

    def zero_grad(self):
        """Reset the accumulated gradients"""
        self.grad.zero_()

    def add_grad(self, grad_tensor):
        """Add the gradient of a batch"""
        if self.grad.shape != grad_tensor.shape:
            raise ValueError(self.grad.shape, grad_tensor.shape)
        self.grad.add_(grad_tensor)

    def cuda(self):
        """Transfert the tensors to the cuda device"""
        self.data = self.data.cuda()
        self.grad = self.grad.cuda()

    def __repr__(self):
        return "{}: data shape {}".format(self.name, self.data.shape)


class Module(object):
    """Base class for a module.
    """
    def __init__(self):
        super(Module, self).__init__()
        self.params = {}
        self.activations = None
        self.outputs = None

    def parameters(self):
        """Return the parameters list of the module"""
        return [] if len(self.params) == 0 else list(self.params.values())

    def cuda(self):
        for p in self.parameters():
            p.cuda()

    def forward(self, *input):
        """Do the forward pass of backprop algo. Subclasses must implement this method.
        :argument *input the input tensor(s)
        :return the output tensor(s)"""
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        """Do the backward pass of backprop algo. Subclasses must implement this method.
        :argument *gradwrtoutput tensor(s) containing the gradient(s) wrt the output of this module
        :return tensor(s) containing the gradient wrt the input(s) of this module"""
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ReLU(Module):
    """ReLU module"""
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        output = input.clamp(min=0)
        self.outputs = output
        return output

    def backward(self, gradwrtoutput):
        g_output = (gradwrtoutput * self.outputs.sign())
        return g_output


class Tanh(Module):
    """Tanh module"""
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, input):
        self.outputs = input.tanh()
        return self.outputs

    def backward(self, gradwrtoutput):
        g_output = (gradwrtoutput * (1 - self.outputs**2))
        return g_output


class Linear(Module):
    """Linear (dense) module.
    :argument input_dim must match the dimension of the input tensor
    :argument output_dim the number of units (neurons) of the layer
    :argument w_init a tensor of size(input_dim, output_dim) containing the initialization of the weights,
                if None, the weights are initilialized according Xavier method
    :argument b_init a tensor of size(output_dim) if None --> init according Xavier method"""
    def __init__(self, input_dim, output_dim, w_init=None, b_init=None):
        super(Linear, self).__init__()
        self.nb_units = output_dim
        self.input_dims = input_dim

        std = math.sqrt(2.0 / (output_dim + input_dim))

        ws = torch.zeros(output_dim, input_dim).normal_(0, std) if w_init is None else w_init
        self.params['w'] = Parameter(ws, name=self.__repr__() + " w")

        bs = torch.zeros(output_dim).uniform_(-std, std) if b_init is None else b_init
        self.params['b'] = Parameter(bs, name=self.__repr__() + " b")

    def forward(self, input):
        self.activations = input
        out = (input @ self.params['w'].data.t()) + self.params['b'].data
        return out

    def backward(self, gradwrtoutput):
        self.params['b'].add_grad(gradwrtoutput.sum(0))
        self.params['w'].add_grad(gradwrtoutput.t() @ self.activations)
        g_output = gradwrtoutput @ self.params['w'].data
        return g_output

    def __repr__(self):
        return "Linear({}, {})".format(self.input_dims, self.nb_units)


class Sequential(Module):
    """Sequential module.
    Usage:
        model = Sequential(
            Linear(2, 128),
            ReLU(),
            Linear(128, 1),
            Tanh()
        )
    """
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.layers = modules

    def __repr__(self):
        return "Sequential(\n  {})".format(",\n  ".join([str(l) for l in self.layers]))

    def forward(self , *input):
        """Do the chain of forward calls on all children modules"""
        out = input[0]
        for l in self.layers:
            out = l.forward(out)
        return out

    def backward(self, *gradwrtoutput):
        """Do the chain of backward calls on all children modules"""
        gradout = gradwrtoutput[0]
        for l in reversed(self.layers):
            gradout = l.backward(gradout)
        return gradout
            
    def parameters(self):
        """Return the paramters of all children modules"""
        params = []
        for m in self.layers:
            p = m.parameters()
            if p is not None:
                params += p
        return params
