#from .debug import log

class MSELoss(object):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_hat, target):
        return ((y_hat-target)**2).mean()

    def backward(self, y_hat, target):
        return 2.0*(y_hat-target)
