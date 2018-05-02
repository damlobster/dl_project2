class SGD(object):
    """Standard SGD optimizer"""
    def __init__(self, parameters, lr=0.01):
        self.lr = lr
        self.parameters = parameters

    def zero_grad(self):
        """Reset the gradients tensors of all parameters"""
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        """Do a step of optimization"""
        for p in self.parameters:
            p.data.add_(-self.lr*p.grad)