import math
import torch
from torch.optim.optimizer import Optimizer

r"""
This script is modified from AdaBelief optimizer script, which is copyrighted by Juntang Zhuang(https://github.com/juntang-zhuang/Adabelief-Optimizer/tree/update_0.2.0)
"""

version_higher = (torch.__version__ >= "1.5.0")

class SGD(Optimizer):

    def __init__(self, params, lr=1e-3, weight_decay=1e-16):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(SGD, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)


    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # State initialization
                state['step'] = 0


    def step(self, closure=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'SGD does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0

                # perform weight decay
                p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                state['step'] += 1

                step_size = group['lr']

                # update
                p.data.add_(grad, alpha=-step_size)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half() 

        return loss

