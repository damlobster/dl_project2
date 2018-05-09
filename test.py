import torch
from simplenet import modules, criterions, optimizers
from datasets_gen import generate_disk_dataset
from simplenet.training import ModelTrainer


def main():
    cuda = False

    #**************************************************************************************
    # SimpleNet: our implementation
    model = modules.Sequential(
        modules.Linear(2, 128),
        modules.ReLU(),
        modules.Linear(128, 1),
        modules.Tanh()
    )

    train_input, train_target, test_input, test_target = generate_disk_dataset(1000, 1000, cuda)
    mt = ModelTrainer(model, criterions.MSELoss(), optimizers.SGD(model.parameters(), lr=0.01),
                      y_hat_fun=torch.sign)

    print("\n                     Train SimpleNet model on toy dataset")
    mt.fit((train_input, train_target), (test_input, test_target), epochs=250, batch_size=100, verbose=10)

    mt.plot_training("SimpleNet learning curves", avg_w_size=5, filename="simplenet_learning_curves.png")

    #**************************************************************************************
    # PyTorch
    from torch import nn, optim, autograd
    model = nn.Sequential(
        nn.Linear(2, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Tanh()
    )

    train_input = autograd.Variable(train_input)
    train_target = autograd.Variable(train_target)
    test_input = autograd.Variable(test_input)
    test_target = autograd.Variable(test_target)

    mt = ModelTrainer(model, nn.MSELoss(), optim.SGD(model.parameters(), lr=0.01),
                      y_hat_fun=torch.sign, pytorch_model=True)

    print("\n\n                      Training PyTorch model on toy dataset")
    mt.fit((train_input, train_target), (test_input, test_target), epochs=250, batch_size=100, verbose=10)

    mt.plot_training("PyTorch learning curves", avg_w_size=5, filename="pytorch_learning_curves.png")

    print('\n\nTraining finished, see plots in current directory (*.png files)\n\n')


if __name__ == "__main__":
    main()
