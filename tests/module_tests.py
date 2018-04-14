import unittest

from simplenet import modules
import torch

class ModulesTest(unittest.TestCase):

    def setUp(self):
        n = 1000
        self.input = torch.randn(n, 99, 100)
        self.out_grad = torch.randn(1, 99, 100)

    def test_relu_forward(self):
        tmodel = torch.nn.ReLU()
        tout = tmodel.forward(self.input)

        model = modules.ReLU()
        out = model(self.input)
        assert(torch.equal(model.activations[0], self.input))
        assert(torch.equal(out, tout.data))

    def test_relu_backward(self):
        tmodel = torch.nn.ReLU()
        tout = tmodel.forward(self.input)

        model = modules.ReLU()
        out = model(self.input)
        assert(torch.equal(model.activations[0], self.input))
        assert(torch.equal(out, tout.data))

    def test_tanh_forward(self):
        tmodel = torch.nn.Tanh()
        tout = tmodel.forward(self.input)

        model = modules.Tanh()
        out = model(self.input)
        assert(torch.equal(model.activations[0], self.input))
        assert(torch.equal(out, tout))

    def test_tanh_backward(self):
        tmodel = torch.nn.Tanh()
        tout = tmodel.forward(self.input)

        model = modules.Tanh()
        out = model(self.input)
        assert(torch.equal(model.activations[0], self.input))
        assert(torch.equal(out, tout))

