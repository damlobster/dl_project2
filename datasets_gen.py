import math
import torch

def generate_disk_dataset(samples_train=1000, sample_test=1000):
    train_input, train_target = _generate_disc_set(samples_train)
    test_input, test_target = _generate_disc_set(sample_test)

    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    return train_input, train_target, test_input, test_target

def _generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(-1, 1)
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().float().view(-1, 1)

    return input, target
