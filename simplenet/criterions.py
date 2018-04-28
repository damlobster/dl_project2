class Criterion(object):
    """Base class for loss functions"""
    def __init__(self):
        super(Criterion, self).__init__()
        self.y_hat = None
        self.target = None

    def cuda(self):
        # for pytorch api compatibility
        return

    def forward(self, y_hat, target):
        """Compute the loss"""
        raise NotImplementedError

    def backward(self):
        """Compute the gradient wrt to the output of the last layer"""
        raise NotImplemented


class MSELoss(Criterion):
    """Implement the MSELoss"""
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_hat, target):
        self.y_hat = y_hat
        self.target = target
        return ((y_hat-target)**2).mean()

    def backward(self):
        return 2.0*(self.y_hat-self.target)
