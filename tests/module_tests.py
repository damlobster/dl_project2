import unittest

from simplenet.modules import *
from simplenet.criterions import MSELoss
from simplenet.training import ModelTrainer
from simplenet.optimizers import SGD

import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable

from datasets_gen import generate_disk_dataset


class ModulesTest(unittest.TestCase):

    def setUp(self):
        self.n = 10000
        self.input_dims = 100
        self.input = torch.randn(self.n, self.input_dims)
        self.gradwrtout = torch.randn(self.n, self.input_dims)

    def torch_eval_f_backward(self, f, gradwrtout):
        tin = Variable(self.input, requires_grad=True)
        tout = f(tin)
        tout.backward(gradwrtout)
        grad = tin.grad.data.sum(0)
        return grad

    def test_relu_forward(self):
        tout = F.relu(torch.autograd.Variable(self.input))

        model = ReLU()
        out = model.forward(self.input)
        assert(torch.equal(model.outputs, out))
        assert(torch.equal(out, tout.data))

    def test_relu_backward(self):
        relu = ReLU()
        relu.forward(self.input)
        grad = relu.backward(self.gradwrtout).sum(0)
        tgrad = self.torch_eval_f_backward(F.relu, self.gradwrtout)
        assert(torch.equal(grad, tgrad))

    def test_tanh_forward(self):
        tmodel = torch.nn.Tanh()
        tout = tmodel.forward(self.input)

        model = Tanh()
        out = model.forward(self.input)
        assert(torch.equal(model.outputs, out))
        assert(torch.equal(out, tout))

    def test_tanh_backward(self):
        tanh = Tanh()
        tanh.forward(self.input)
        grad = tanh.backward(self.gradwrtout).sum(0)
        tgrad = self.torch_eval_f_backward(F.tanh, self.gradwrtout)
        assert(torch.equal(torch.round(grad*100), torch.round(tgrad*100)))

    def test_linear_forward(self):
        dims = (self.input_dims,10)
        tmodel = torch.nn.Linear(*dims)
        #copy pytorch parameters initialization
        tparams = tuple(p.data.clone() for p in tmodel.parameters())
        tout = tmodel.forward(torch.autograd.Variable(self.input))

        model = Linear(*dims, w_init=tparams[0], b_init=tparams[1])
        out = model.forward(self.input)
        assert(torch.equal(model.activations, self.input))
        assert(torch.equal(out, tout.data))

    def test_linear_backward(self):
        gradwrtout = torch.randn(self.n, 10)
        weights = torch.randn(10, self.input_dims)
        bias = torch.randn(10)
        model = Linear(self.input_dims, 10, w_init=weights, b_init=bias)
        model.activations = self.input
        gradwrtin = model.backward(gradwrtout)

        ws = Variable(weights, requires_grad=True)
        bs = Variable(bias, requires_grad=True)
        lin_f = lambda input: F.linear(input, ws, bs)
        tgrad = self.torch_eval_f_backward(lin_f, gradwrtout)

        assert(torch.equal(gradwrtin.sum(0), tgrad))
        assert (torch.equal(model.params['w'].grad, ws.grad.data))
        assert (torch.equal(model.params['b'].grad, bs.grad.data))


    def test_disk_shallow(self):
        model = Sequential(
            Linear(2, 128),
            ReLU(),
            Linear(128, 1),
            Tanh()
        )

        train_input, train_target, test_input, test_target = generate_disk_dataset()

        mt = ModelTrainer(model, MSELoss(), SGD(model.parameters(), lr=0.01))

        mt.fit((train_input, train_target), epochs=250, batch_size=100, verbose=0)

        y_hat = mt.predict(test_input)
        acc = compute_accuracy(test_target, y_hat)
        print("test accuracy=", acc)
        assert(acc > 0.95)

    def test_model_trainer(self):
        model = Sequential(
            Linear(2, 128),
            ReLU(),
            Linear(128, 1),
            Tanh()
        )

        train_input, train_target, test_input, test_target = generate_disk_dataset()

        mt = ModelTrainer(model, MSELoss(), SGD(model.parameters(), lr=0.01), y_hat_fun=torch.sign)

        mt.fit((train_input, train_target), (test_input, test_target), epochs=25, batch_size=100, verbose=0)

        # pytorch test
        from torch import nn, optim, autograd
        model = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

        train_input, train_target, test_input, test_target = generate_disk_dataset()
        train_input = autograd.Variable(train_input)
        train_target = autograd.Variable(train_target)
        test_input = autograd.Variable(test_input)
        test_target = autograd.Variable(test_target)

        mt = ModelTrainer(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.01),
                          y_hat_fun=torch.sign, pytorch_model=True)

        mt.fit((train_input, train_target), (test_input, test_target), epochs=25, batch_size=100, verbose=0)


def compute_accuracy(target, y_hat):
    nb_correct = 0
    y = torch.sign(y_hat)
    for i in range(target.shape[0]):
        nb_correct += 1 if target[i].equal(y[i]) else 0

    return nb_correct / target.shape[0]