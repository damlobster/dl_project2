import torch
#from .debug import log

class MSELoss(object):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_hat, target):
        return ((y_hat-target)**2).mean()

    def backward(self, y_hat, target):
        return 2.0*(y_hat-target)

class SGD(object):
    def __init__(self, lr=0.01):
        self.lr = lr

    def do_step(self, param):
        param.data.add_(-self.lr*param.grad)

class ModelTrainer(object):
    def __init__(self, model, criterion):
        super(ModelTrainer).__init__()
        self.model = model
        self.criterion = criterion

    def fit(self, input, target, optimizer, epochs=50, batch_size=None, validation_split=0.0, verbose=1):
        N = input.shape[0]

        history = {'loss': []}
        for epoch in range(epochs):
            batches = torch.randperm(N)
            if batch_size is not None:
                batches = batches.split(batch_size)
            else:
                batches = [batches]

            for batch in batches:
                self.model.zero_grads()
                inp = torch.index_select(input, 0, batch)
                tar = torch.index_select(target, 0, batch)
                #print(inp, tar)
                modout  = self.model.forward(inp)
                loss = self.criterion.forward(modout, tar)
                history['loss'] += [loss]
                gradcrit = self.criterion.backward(modout, tar)
                self.model.backward(gradcrit)
                for p in self.model.get_params():
                    optimizer.do_step(p)

            if verbose>0 and epoch%verbose==0:
                print("epoch: {}, loss:{}".format(epoch, loss))

        return history

    def predict(self, input):
        return self.model.forward(input)