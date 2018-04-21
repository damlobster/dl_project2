class SGD(object):
    def __init__(self, lr=0.01):
        self.lr = lr

    def do_step(self, param):
        param.data.add_(-self.lr*param.grad)