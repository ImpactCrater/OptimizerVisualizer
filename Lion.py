import math
import torch
from torch.optim.optimizer import Optimizer

r"""
This script is modified from AdaBelief optimizer script, which is copyrighted by Juntang Zhuang(https://github.com/juntang-zhuang/Adabelief-Optimizer/tree/update_0.2.0)
"""

version_higher = (torch.__version__ >= "1.5.0")

class Lion(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-16, weight_decay=1e-16):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, buffer=[[None, None, None] for _ in range(10)])
        super(Lion, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(Lion, self).__setstate__(state)


    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # State initialization
                state['step'] = 0
                state['ema'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)


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
                        'Lion does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['ema'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)

                # perform weight decay
                p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                # get current state variable
                ema = state['ema']

                state['step'] += 1
                #bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update moment running average
                modifiedGrad = ema.mul(beta1).add(grad, alpha=1 - beta1)
                ema.mul_(beta2).add_(grad, alpha=1 - beta2)

                #step_size = group['lr'] / bias_correction1
                step_size = group['lr']

                # update
                p.data.add_(torch.sign(modifiedGrad), alpha=-step_size)

                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half() 

        return loss

