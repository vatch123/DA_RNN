import torch
import torch.nn as nn

class CustomLoss1(nn.Module):
    """
    A class which defines a custom loss function
    """
    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, output, labels):
        """
        The loss function implemented is e^(3*abs(y_pred - y_true))
        """
        x = self.l1loss(output, labels)
        loss = torch.mean(torch.expm1(torch.mul(x, 3)))
        # print("X: ",x, " Loss: ", loss)
        return loss


class CustomLoss2(nn.Module):
    """
    A class which defines a custom loss function
    """
    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, output, labels):
        """
        The loss function implemented is e^(1.5*(y_pred - y_true)^2)
        """
        x = self.l1loss(output, labels)
        loss = torch.mean(torch.expm1(torch.mul(x ** 2, 1.5)))
        # print("X: ",x, " Loss: ", loss)
        return loss


class CustomLoss3(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2loss = nn.MSELoss()

    def forward(self, output, labels):
        """
        The loss function implemented is 3*(y_pred - y_true)^2
        """
        x = self.l2loss(output, labels)
        loss = torch.mul(x, 3)
        # print("X: ",x, " Loss: ", loss)
        return loss
