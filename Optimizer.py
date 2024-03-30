import math
import torch
from torch.optim import Optimizer

class FRGD(Optimizer):
    """
    Implements Stochastic Fletcher-Reeves (FR) Conjugate Gradient optimization algorithm with weight decay.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        weight_decay (float): weight decay coefficient
    """

    def __init__(self, params, lr, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(FRGD, self).__init__(params, defaults)

        for group in self.param_groups:
            for param in group['params']:
                param_state = self.state[param]
                param_state['conjugate_direction'] = torch.zeros_like(param.data)
                param_state['conjugate_gradient'] = torch.zeros_like(param.data)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                d_p = param.grad.data
                d_p.add_(param.data, alpha=weight_decay)  # 添加权重衰减项

                param_state = self.state[param]
                conjugate_direction = param_state['conjugate_direction']
                conjugate_gradient = param_state['conjugate_gradient']
                weight_decay = group['weight_decay']

                if torch.norm(conjugate_gradient) == 0:
                    # 如果共轭梯度为零，则更新为负梯度
                    conjugate_direction = -d_p
                else:
                    beta = min((torch.norm(d_p, p=2)**2 / torch.norm(conjugate_gradient, p=2)**2).item(), 0.999)
                    conjugate_direction = -d_p + beta * conjugate_direction

                # 更新参数
                param.data.add_(conjugate_direction, alpha=group['lr'])

                # 更新共轭梯度
                conjugate_gradient = d_p.clone()

                param_state['conjugate_direction'] = conjugate_direction
                param_state['conjugate_gradient'] = conjugate_gradient

        return loss
class SGDM(Optimizer):

    def __init__(self, params, lr, momentum=0.9,weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay,momentum=momentum)
        super(SGDM, self).__init__(params, defaults)

        for group in self.param_groups:
            for param in group['params']:
                param_state = self.state[param]
                param_state['conjugate_direction'] = torch.zeros_like(param.data)
                param_state['conjugate_gradient'] = torch.zeros_like(param.data)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue

                d_p = param.grad.data
                d_p.add_(param.data, alpha=weight_decay)  # 添加权重衰减项
                param_state = self.state[param]
                conjugate_direction = param_state['conjugate_direction']
                conjugate_gradient = param_state['conjugate_gradient']


                conjugate_direction = -d_p + momentum * conjugate_direction

                # 更新参数
                param.data.add_(conjugate_direction, alpha=group['lr'])

                # 更新共轭梯度
                conjugate_gradient = d_p.clone()

                param_state['conjugate_direction'] = conjugate_direction
                param_state['conjugate_gradient'] = conjugate_gradient

        return loss

class SGDAMD(Optimizer):
    """
    Implements Stochastic Fletcher-Reeves (FR) Conjugate Gradient optimization algorithm with weight decay.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        weight_decay (float): weight decay coefficient
    """

    def __init__(self, params, lr, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SGDAMD, self).__init__(params, defaults)

        for group in self.param_groups:
            for param in group['params']:
                param_state = self.state[param]
                param_state['conjugate_direction'] = torch.zeros_like(param.data)
                param_state['conjugate_gradient'] = torch.zeros_like(param.data)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for param in group['params']:
                if param.grad is None:
                    continue
                d_p = param.grad.data
                d_p.add_(param.data, alpha=weight_decay)  # 添加权重衰减项
                param_state = self.state[param]
                conjugate_direction = param_state['conjugate_direction']
                conjugate_gradient = param_state['conjugate_gradient']
                weight_decay = group['weight_decay']
                D_g = d_p - conjugate_gradient
                K = 0.6 * (torch.norm(d_p, p=2).item())
                D = (torch.norm(D_g, p=2).item())
                if D > K:
                    gamma = 0
                else:
                    gamma = 1
                y = gamma * D_g + ((1 - gamma) * D_g * K) / (D + 0.00001)


                beta = ((torch.norm(d_p, p=2) ** 2) / (
                            torch.norm(d_p, p=2) ** 2 + 0.5 * torch.norm(conjugate_gradient, p=2) ** 2)).item()
                beta = min(abs(beta), 0.999)

                if beta == 0.999:
                    k_d = 0.01
                else:
                    k_d = 0.000001

                conjugate_direction = d_p + beta * conjugate_direction + k_d * y

                # 更新参数
                param.data.add_(conjugate_direction, alpha=-group['lr'])

                # 更新共轭梯度
                conjugate_gradient = d_p.clone()

                param_state['conjugate_direction'] = conjugate_direction
                param_state['conjugate_gradient'] = conjugate_gradient

        return loss

class Adam(Optimizer):
    """
    Implements Modified Adam optimization algorithm with weight decay.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        betas (Tuple[float, float], optional): coefficients for computing running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve numerical stability
        weight_decay (float, optional): weight decay coefficient
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

        for group in self.param_groups:
            for param in group['params']:
                param_state = self.state[param]
                param_state['exp_avg'] = torch.zeros_like(param.data)
                param_state['exp_avg_sq'] = torch.zeros_like(param.data)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for param in group['params']:
                if param.grad is None:
                    continue

                d_p = param.grad.data
                d_p.add_(param.data, alpha=weight_decay)  # 添加权重衰减项
                param_state = self.state[param]
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add(eps)
                step_size = group['lr']

                param.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class RAdam(Optimizer):
    """
    Implements Rectified Adam (RAdam) optimization algorithm.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        betas (Tuple[float, float], optional): coefficients for computing running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve numerical stability
        weight_decay (float, optional): weight decay coefficient
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for param in group['params']:
                param_state = self.state[param]
                param_state['step'] = 0
                param_state['exp_avg'] = torch.zeros_like(param.data)
                param_state['exp_avg_sq'] = torch.zeros_like(param.data)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for param in group['params']:
                if param.grad is None:
                    continue

                d_p = param.grad.data
                d_p.add_(param.data, alpha=weight_decay)  # 添加权重衰减项
                param_state = self.state[param]

                step = param_state['step']
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']

                step += 1
                beta2_t = beta2 ** step
                beta1_t = beta1 * (1 - 0.5 * (0.96 ** (step * 0.5)))

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)

                # Compute max length of the approximated uncentered variance
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * step * (beta2 ** step) / (1 - (beta2 ** step))

                rho_threshold = 2 / (1 - beta2) - 4
                if rho_t > rho_threshold:
                    r_hat = math.sqrt((1 - beta2_t) / (exp_avg_sq / (1 - beta2) + eps))
                    r = r_hat * math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    step_size = group['lr'] * r
                else:
                    step_size = group['lr']

                # Update parameters
                param.data.addcdiv_(exp_avg, exp_avg_sq.sqrt() + eps, value=-step_size)

        return loss

class SGDMD(Optimizer):
    """
    Implements Stochastic Fletcher-Reeves (FR) Conjugate Gradient optimization algorithm with weight decay.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate
        weight_decay (float): weight decay coefficient
    """

    def __init__(self, params, lr, momentum=0.9,weight_decay=0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay,momentum=momentum)
        super(SGDMD, self).__init__(params, defaults)

        for group in self.param_groups:
            for param in group['params']:
                param_state = self.state[param]
                param_state['conjugate_direction'] = torch.zeros_like(param.data)
                param_state['conjugate_gradient'] = torch.zeros_like(param.data)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue

                d_p = param.grad.data
                d_p.add_(param.data, alpha=weight_decay)  # 添加权重衰减项
                param_state = self.state[param]
                conjugate_direction = param_state['conjugate_direction']
                conjugate_gradient = param_state['conjugate_gradient']

                if torch.norm(conjugate_gradient) == 0:
                    conjugate_direction = -d_p
                else:
                    y=d_p-conjugate_gradient

                    conjugate_direction = y + momentum * conjugate_direction
                    conjugate_direction = 0.1*conjugate_direction + d_p

                # 更新参数
                param.data.add_(conjugate_direction, alpha=-group['lr'])

                # 更新共轭梯度
                conjugate_gradient = d_p.clone()

                param_state['conjugate_direction'] = conjugate_direction
                param_state['conjugate_gradient'] = conjugate_gradient

        return loss

