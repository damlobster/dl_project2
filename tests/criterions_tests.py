import unittest

from simplenet import criterions
import torch
import torch.nn.functional as F


class CriterionsTest(unittest.TestCase):

    def setUp(self):
        n = 100000
        self.targets = torch.randn(n, 1)
        self.y_hat = torch.randn(n, 1)

    def test_mseloss_forward(self):
        tloss = F.mse_loss(self.y_hat, self.targets)

        model = criterions.MSELoss()
        out = model.forward(self.y_hat, self.targets)
        print(out, tloss.data[0])
        assert(out == tloss.data[0])
