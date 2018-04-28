import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


class ModelTrainer(object):
    """Utility class to train models. Compatible with SimpleNet and PyTorch models.
    Usage:
        model = ...
        train_data = (x, y)
        test_data = (...)
        mt = ModelTrainer(model, MSELoss(), SGD(model.parameters()))
        mt.fit(train_data, test_data, epochs=250, batch_size=100, verbose=10)
        mt.plot_training("Learning curves")
    """
    def __init__(self, model, criterion, optimizer, y_hat_fun=lambda y: y,
                 criterion_fun=lambda x, y:(x, y), pytorch_model=False):
        """Initialize a ModelTrainer.
        :argument model a SimpleNet or PyTorch model
        :argument criterion the loss function, see criterion.py
        :argument optimizer the optimization algo to use, see optimizers.py
        :argument y_hat_fun function to process the output of the last layer
        :argument pytorch_model set to True if the model is a PyTorch one"""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.y_hat_fun = y_hat_fun
        self.criterion_fun = criterion_fun
        self.pytorch_model = pytorch_model
        self.best_model = None

    def fit(self, train_data, validation_data=None, epochs=25, batch_size=None, verbose=1):
        """Fit the model on the training data.
        :argument train_data (x_train, y_train)
        :argument test_data (x_validation, y_validation)
        :argument epochs nb of epochs to train
        :argument batch_size the mini-batchs size
        :argument verbose print the current loss and accuracies if current_epoch%verbose == 0"""
        self.history = History()

        x_train, y_train = train_data
        N = x_train.data.shape[0] if self.pytorch_model else x_train.shape[0]

        if x_train.is_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        if self.pytorch_model:
            self.model.train(True)

        for epoch in range(epochs):

            idxs = torch.randperm(N)
            if(x_train.is_cuda):
                idxs = idxs.cuda()
            idxs = [idxs] if batch_size is None else idxs.split(batch_size)

            train_correct = 0
            train_loss = 0
            for batch in idxs:
                self.optimizer.zero_grad()

                y_hat = self.model.forward(x_train[batch])
                loss = self.criterion.forward(*self.criterion_fun(y_hat, y_train[batch]))

                if self.pytorch_model:
                    loss.backward()
                else:
                    gradwrtloss = self.criterion.backward()
                    self.model.backward(gradwrtloss)

                self.optimizer.step()

                if self.pytorch_model:
                    train_correct += (self.y_hat_fun(y_hat) == y_train[batch]).sum().data[0]
                else:
                    train_correct += (self.y_hat_fun(y_hat) == y_train[batch]).sum()

                train_loss += loss

            val_loss = np.nan
            val_acc = np.nan

            if validation_data is not None:
                #Disable training mode
                if self.pytorch_model:
                    self.model.train(False)

                x_test, y_test = validation_data
                y_hat = self.model.forward(x_test)
                if self.pytorch_model:
                    val_loss = self.criterion(*self.criterion_fun(y_hat, y_test)).data[0]/len(x_test)
                    val_acc = (self.y_hat_fun(y_hat).data==y_test.data).float().sum()/len(x_test)
                else:
                    val_loss = self.criterion.forward(*self.criterion_fun(y_hat, y_test))/len(x_test)
                    val_acc = (self.y_hat_fun(y_hat)==y_test).float().sum()/len(x_test)



            if self.pytorch_model:
                self.history.add([
                    train_loss.data[0]/x_train.size()[0],
                    train_correct/x_train.size()[0],
                    val_loss,
                    val_acc
                ])
            else:
                self.history.add([
                    train_loss/x_train.shape[0],
                    train_correct/x_train.shape[0],
                    val_loss,
                    val_acc
                ])

            if (val_acc >= self.history.get_best_val_acc()):
                self.best_model = copy.deepcopy(self.model)

            if verbose != 0 and epoch%verbose == 0:
                if epoch == 0:
                    print("******************************** Train log ************************************")
                print(self.history.get_last().to_string(col_space=15, header=epoch==0, formatters=History.formatters))

        if verbose != 0:
            print("Best validation loss:")
            print(self.history.get_best().to_string(col_space=15, header=epoch==0, formatters=History.formatters))
            print("*******************************************************************************")

        return self.history

    def get_best_model(self):
        return self.best_model

    def plot_training(self, title, avg_w_size=20, filename=None):
        if self.history is None:
            print("Train model first!")
        else:
            self.history.plot(title, avg_w_size)
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename)

    def predict(self, input):
        if self.pytorch_model:
            self.model.train(False)
        return self.model.forward(input)


class History(object):
    """Hold the training history"""
    def __init__(self):
        self.hist = pd.DataFrame(columns=['train loss', 'train acc', 'val loss', 'val acc'])

    formatters = {
        'train loss': "{:0<0.05f}".format,
        'train acc': "{:0.3f}".format,
        'val loss': "{:0<0.05f}".format,
        'val acc': "{:0.3f}".format}

    def add(self, new_epoch):
        self.hist.loc[len(self.hist)] = new_epoch

    def get_last(self):
        return self.hist.tail(1)

    def get_best(self, n=1):
        return self.hist.sort_values('val loss').head(n)

    def get_best_val_acc(self):
        return self.hist.sort_values('val acc', ascending=False).head(1)['val acc'].values[0]

    def get_best_epochs_nb(self, n=1):
        return self.hist.sort_values('val loss').head(n).index.tolist()

    def get_hist(self):
        return self.hist

    def plot(self, title, avg_w_size=20):
        colors = ['C0', 'C1']
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
        fig.suptitle(title)
        self.hist[['train loss', 'val loss']].ewm(span=avg_w_size).mean().plot(ax=ax1, color=colors)
        self.hist[['train loss', 'val loss']].plot(ax=ax1, alpha=0.4, color=colors, legend=False)
        self.hist[['train acc', 'val acc']].ewm(span=avg_w_size).mean().plot(ax=ax2, color=colors)
        self.hist[['train acc', 'val acc']].plot(ax=ax2, alpha=0.4, color=colors, legend=False)
        ax1.set_ylabel('categorical cross entropy')
        ax1.set_xlabel('epochs')
        ax1.set_yscale('log')
        ax1.grid(color='0.8', linewidth=0.5, ls='--')
        ax2.set_ylabel('accuracy [% correct]')
        ax2.set_xlabel('epochs')
        ax2.grid(color='0.8', linewidth=0.5, ls='--')
        return fig
