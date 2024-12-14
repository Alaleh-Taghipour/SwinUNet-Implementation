import torch
from collections import defaultdict


class ASAM:
    """
    Implements the Adaptive Sharpness-Aware Minimization (ASAM) optimizer, an extension of SAM designed to adaptively scale gradients.

    Args:
        optimizer (torch.optim.Optimizer): The base optimizer to use (e.g., SGD, Adam).
        model (torch.nn.Module): The neural network model being optimized.
        rho (float, optional): The radius of the neighborhood to consider for sharpness minimization. Default is 0.5.
        eta (float, optional): Adaptive scaling factor for the weights. Default is 0.01.
    """

    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)  # Stores state information for each parameter

    @torch.no_grad()
    def ascent_step(self):
        """
        Performs the ascent step, adjusting the weights to maximize the loss within the rho neighborhood.
        """
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))

        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        """
        Performs the descent step, restoring parameters to their original values and updating them using the base optimizer.
        """
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    """
    Implements the Sharpness-Aware Minimization (SAM) optimizer, which minimizes sharpness by perturbing parameters in the direction of the gradient.

    Inherits from ASAM but does not include adaptive scaling (eta).

    Args:
        optimizer (torch.optim.Optimizer): The base optimizer to use (e.g., SGD, Adam).
        model (torch.nn.Module): The neural network model being optimized.
        rho (float, optional): The radius of the neighborhood to consider for sharpness minimization. Default is 0.5.
    """

    @torch.no_grad()
    def ascent_step(self):
        """
        Performs the ascent step, adjusting weights to maximize the loss within the rho neighborhood.
        """
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()
