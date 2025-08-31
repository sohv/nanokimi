"""
Muon Optimizer Implementation for nanoKimi

Based on the Muon optimizer described in Kimi-K2 papers.
Combines momentum with adaptive learning rates for better convergence.
"""

import torch
import torch.optim as optimizer
from typing import Any, Dict, Optional


class Muon(optimizer.Optimizer):
    """
    Muon optimizer: A momentum-based optimizer with adaptive learning rates
    
    This optimizer combines the benefits of momentum with adaptive learning rate
    scaling, designed specifically for large language model training.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        momentum: momentum factor (default: 0.9)
        weight_decay: weight decay (L2 penalty) (default: 0.01)
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        backend: backend to use ('torch' or 'triton') (default: 'torch')
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        backend: str = 'torch'
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            backend=backend
        )
        super(Muon, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                param_state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(momentum).add_(grad, alpha=1 - momentum)
                exp_avg_sq.mul_(momentum).addcmul_(grad, grad, value=1 - momentum)
                
                # Bias correction
                step = param_state['step']
                bias_correction1 = 1 - momentum ** step
                bias_correction2 = 1 - momentum ** step
                
                # Compute the denominator
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                
                # Compute the step size
                step_size = lr / bias_correction1
                
                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()


def create_muon_optimizer(model, config):
    """Create Muon optimizer with the given configuration"""
    return Muon(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        eps=config['eps'],
        backend=config.get('backend', 'torch')
    )
