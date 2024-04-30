import numpy as np
import torch


class Optimizer:

    def __init__(
            self,
            alpha_up=1.015,
            alpha_down=0.9,
            lr_lb=0,
            lr_ub=1e+5,
            verbose=False
    ):
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down
        self.lr_lb = lr_lb
        self.lr_ub = lr_ub
        self.verbose = verbose

        self.optimizers = []
        self.parameters = []
        self.cursor = 0
        self.last_loss = np.inf

    def add(self, param, lr):
        params = [{
            'params': param,
            'lr': lr
        }]
        self.parameters.append(params[0])
        self.optimizers.append(torch.optim.Adam(params, lr=lr))

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self, loss):

        # Update learning rate
        if self.cursor >= len(self.parameters):
            return
        previous_cursor = (self.cursor + len(self.parameters) - 1) % len(self.parameters)
        if loss <= self.last_loss:
            factor = self.alpha_up
        else:
            factor = self.alpha_down
        lr = self.parameters[previous_cursor]['lr']
        lr = float(np.clip(lr * factor, self.lr_lb, self.lr_ub))
        self.parameters[previous_cursor]['lr'] = lr
        for group in self.optimizers[previous_cursor].param_groups:
            group['lr'] = lr
        self.last_loss = loss

        self.optimizers[self.cursor].step()

        self.cursor = (self.cursor + 1) % len(self.parameters)
