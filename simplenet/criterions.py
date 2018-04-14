from operator import mul
from functools import reduce

class MSELoss(object):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_hat, target):
        return ((target-y_hat)**2).mean()

    def backward(self, y_hat, target):
        return 2.0*(target-y_hat).mean()