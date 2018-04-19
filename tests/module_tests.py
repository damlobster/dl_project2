import unittest

from simplenet import modules
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable


class ModulesTest(unittest.TestCase):

    def setUp(self):
        self.n = 10000
        self.input_dims = 100
        self.input = torch.randn(self.n, self.input_dims)
        self.gradwrtout = torch.randn(self.n, self.input_dims)

        self.addTypeEqualityFunc(torch.FloatTensor, self._tensor_equal)


    def torch_eval_f_backward(self, f, gradwrtout):
        tin = Variable(self.input, requires_grad=True)
        tout = f(tin)
        tout.backward(gradwrtout)
        grad = tin.grad.data.sum(0)
        return grad

    def test_relu_forward(self):
        tout = F.relu(self.input)

        model = modules.ReLU()
        out = model.forward(self.input)
        assert(torch.equal(model.outputs, out))
        assert(torch.equal(out, tout.data))

    def test_relu_backward(self):
        relu = modules.ReLU()
        relu.forward(self.input)
        grad = relu.backward(self.gradwrtout)
        tgrad = self.torch_eval_f_backward(F.relu, self.gradwrtout)
        self.assertEqual(grad, tgrad)

    def test_tanh_forward(self):
        tmodel = torch.nn.Tanh()
        tout = tmodel.forward(self.input)

        model = modules.Tanh()
        out = model.forward(self.input)
        assert(torch.equal(model.outputs, out))
        assert(torch.equal(out, tout))

    def test_tanh_backward(self):
        tanh = modules.Tanh()
        tanh.forward(self.input)
        grad = tanh.backward(self.gradwrtout)
        tgrad = self.torch_eval_f_backward(F.tanh, self.gradwrtout)
        self.assertEqual(grad, tgrad)

    def test_linear_forward(self):
        dims = (self.input_dims,10)
        tmodel = torch.nn.Linear(*dims)
        #copy pytorch parameters initialization
        tparams = tuple(p.data.clone() for p in tmodel.parameters())
        tout = tmodel.forward(torch.autograd.Variable(self.input))

        model = modules.Linear(*dims, w_init=tparams[0], b_init=tparams[1])
        out = model.forward(self.input)
        assert(torch.equal(model.activations, self.input))
        assert(torch.equal(out, tout.data))

    def test_linear_backward(self):
        gradwrtout = torch.randn(self.n, 10)
        weights = torch.randn(10, self.input_dims)
        bias = torch.randn(10)
        model = modules.Linear(self.input_dims, 10, w_init=weights, b_init=bias)
        model.activations = self.input
        gradwrtin = model.backward(gradwrtout)

        ws = Variable(weights, requires_grad=True)
        bs = Variable(bias, requires_grad=True)
        lin_f = lambda input: F.linear(input, ws, bs)
        tgrad = self.torch_eval_f_backward(lin_f, gradwrtout)
        self.assertEqual(gradwrtin, tgrad)

    def _tensor_equal(self, x, y, msg):
        def round(x):
            return (x*100).round() #FIXME ????

        if not torch.equal(x, y):
            if not torch.equal(round(x), round(y)):
                for i in range(x.shape[0]):
                   print(x[i], y[i])
                m = "x!=y" if msg is None else msg
                raise self.failureException(m)
