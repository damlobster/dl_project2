import unittest

from simplenet import modules
import torch
from torch.autograd.variable import Variable

class ModulesTest(unittest.TestCase):

    class Sum(modules.Module):
        def __init__(self):
            super(ModulesTest.Sum, self).__init__()

        def forward(self, input):
            self.activations = input
            return torch.tensor([input.sum()])

        def backward(self, gradwrtoutput):
            return gradwrtoutput * torch.ones(gradwrtoutput.shape)

        def _tensor_equal(self, x, y, msg):
            if not torch.equal(x, y):
                m = "x!=y" if msg is None else msg
                raise self.failureException(msg)

    def setUp(self):
        n = 4
        self.input_dims = (3)
        self.input = torch.randn(n, self.input_dims)

        self.addTypeEqualityFunc(torch.FloatTensor, self._tensor_equal)

    def test_relu_forward(self):
        tmodel = torch.nn.ReLU()
        tout = tmodel.forward(self.input)

        model = modules.ReLU()
        out = model.forward(self.input)
        assert(torch.equal(model.activations, self.input))
        assert(torch.equal(out, tout.data))

    def test_relu_backward(self):
        input = Variable(self.input, requires_grad=True)
        tmodel = torch.nn.ReLU()
        tout = tmodel.forward(input)
        tout = tout.sum()
        tgrad = torch.autograd.grad(tout, input)[0].data

        sum = modules.Sum()
        relu = modules.ReLU()
        out = sum.forward(relu.forward(self.input))
        grad = relu.backward(sum.backward(out))
        self.assertEqual(relu.activations, self.input)
        self.assertEqual(grad, tgrad)

    def test_tanh_forward(self):
        tmodel = torch.nn.Tanh()
        tout = tmodel.forward(self.input)

        model = modules.Tanh()
        out = model.forward(self.input)
        assert(torch.equal(model.activations, self.input))
        assert(torch.equal(out, tout))

    def test_tanh_backward(self):
        tmodel = torch.nn.Tanh()
        tout = tmodel.forward(self.input)

        model = modules.Tanh()
        out = model.forward(self.input)
        assert(torch.equal(model.activations, self.input))
        assert(torch.equal(out, tout))

    def test_linear_forward(self):
        dims = (self.input_dims,10)
        tmodel = torch.nn.Linear(*dims)
        #copy pytorch parameters initialization
        tparams = tuple(p.data.clone() for p in tmodel.parameters())
        tout = tmodel.forward(torch.autograd.Variable(self.input))

        model = modules.Linear(*dims)
        model.params = tparams
        out = model.forward(self.input)
        assert(torch.equal(model.activations, self.input))
        assert(torch.equal(out, tout.data))

    def test_linear_backward(self):
        dims = (self.input_dims,10)
        tmodel = torch.nn.Linear(*dims)
        #copy pytorch parameters initialization
        tparams = tuple(p.data.clone() for p in tmodel.parameters())
        tout = tmodel.f(torch.autograd.Variable(self.input))

        model = modules.Linear(*dims)
        model.params = tparams
        out = model.forward(self.input)
        assert(torch.equal(model.activations, self.input))
        assert(torch.equal(out, tout.data))
