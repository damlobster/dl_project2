import unittest

from simplenet.modules import *
from simplenet import criterions
import math

import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable

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
        tout = F.relu(self.input)

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

    def test_linear_backward_xor(self):
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


    def test_sequential_forward(self):
        model = Sequential(
            Linear(2, 4),
            Tanh(),
            Linear(4, 1),
            Tanh()
        )
        input = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        target = torch.FloatTensor([[-1], [1], [1], [-1]])

        mt = criterions.ModelTrainer(model, criterions.MSELoss())

        mt.fit(input, target, criterions.SGD(lr=0.1), epochs=250, batch_size=1, verbose=0)

        y_hat = mt.predict(input)
        assert((y_hat.sign()).float().equal(target))

    def test_disk_shallow(self):
        model = Sequential(
            Linear(2, 128),
            ReLU(),
            Linear(128, 1),
            Tanh()
        )

        train_input, train_target, test_input, test_target = generate_disk_dataset()

        mt = criterions.ModelTrainer(model, criterions.MSELoss())

        mt.fit(train_input, train_target, criterions.SGD(lr=0.01), epochs=250, batch_size=100, verbose=0)

        y_hat = mt.predict(test_input)
        acc = compute_accuracy(test_target, y_hat)
        print("test accuracy=", acc)
        assert(acc > 0.95)

def compute_accuracy(target, y_hat):
    nb_correct = 0
    y = torch.sign(y_hat)
    for i in range(target.shape[0]):
        nb_correct += 1 if target[i].equal(y[i]) else 0

    return nb_correct / target.shape[0]

def generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().float().view(-1, 1)

    return input, target

def generate_disk_dataset():
    train_input, train_target = generate_disc_set(1000)
    test_input, test_target = generate_disc_set(1000)

    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    return train_input, train_target, test_input, test_target