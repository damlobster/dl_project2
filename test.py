import matplotlib
# needed to run this script on a server without GUI
matplotlib.use('Agg') 

import sys
import time

import torch
from simplenet import modules, criterions, optimizers
from datasets_gen import generate_disk_dataset
from simplenet.training import ModelTrainer


def main(m, samples, batch_size, units):
    # train on GPU if any available
    cuda = torch.cuda.is_available()
    train_input, train_target, test_input, test_target = generate_disk_dataset(samples, 1000, cuda)

    #**************************************************************************************
    # SimpleNet: our implementation
    if m is None or m == 'sn':
        model = modules.Sequential(
            modules.Linear(2, units),
            modules.ReLU(),
            modules.Linear(units, 1),
            modules.Tanh()
        )
        if cuda:
            model.cuda()

        mt = ModelTrainer(model, criterions.MSELoss(), optimizers.SGD(model.parameters(), lr=0.01),
                          y_hat_fun=torch.sign)

        print("\n### Train SimpleNet model on toy dataset ###")
        print(model, "\n")

        start = time.time()
        mt.fit((train_input, train_target), (test_input, test_target), epochs=250, batch_size=batch_size, verbose=10)
        print("Training time = {:0.3f} s".format(time.time()-start))

        mt.plot_training("SimpleNet learning curves", avg_w_size=5, filename="simplenet_learning_curves.png")

    #**************************************************************************************
    # PyTorch
    if m is None or m == 'pt':
        from torch import nn, optim, autograd
        model = nn.Sequential(
            nn.Linear(2, units),
            nn.ReLU(),
            nn.Linear(units, 1),
            nn.Tanh()
        )
        if cuda:
            model.cuda()

        train_input = autograd.Variable(train_input)
        train_target = autograd.Variable(train_target)
        test_input = autograd.Variable(test_input)
        test_target = autograd.Variable(test_target)

        mt = ModelTrainer(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.01),
                          y_hat_fun=torch.sign, pytorch_model=True)

        print("\n\n### Training PyTorch model on toy dataset ###")
        print(model, "\n")

        start = time.time()
        mt.fit((train_input, train_target), (test_input, test_target), epochs=250, batch_size=batch_size, verbose=10)
        print("Training time = {:0.3f} s".format(time.time()-start))

        mt.plot_training("PyTorch learning curves", avg_w_size=5, filename="pytorch_learning_curves.png")

        print('\n\nTraining finished, see plots in current directory (*.png files)\n\n')


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) >= 2 else None
    samples = int(sys.argv[2]) if len(sys.argv) >= 3 else 1000
    batch_size = int(sys.argv[3]) if len(sys.argv) == 4 else 100
    units = int(sys.argv[4]) if len(sys.argv) == 5 else 128
    main(model, samples, batch_size, units)
