import torch

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

def compute_accuracy(target, y_hat):
    nb_correct = 0
    y = torch.sign(y_hat)
    for i in range(target.shape[0]):
        nb_correct += 1 if target[i].equal(y[i]) else 0

    return nb_correct / target.shape[0]