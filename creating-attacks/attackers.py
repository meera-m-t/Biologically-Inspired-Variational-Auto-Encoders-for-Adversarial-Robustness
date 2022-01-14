import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
#torchattacks

class Attacker:
    def __init__(self, clip_max=0.5, clip_min=-0.5):
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        pass

class FGSM(Attacker):
    """
    Fast Gradient Sign Method
    Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy.
    Explaining and Harnessing Adversarial Examples.
    ICLR, 2015
    """
    def __init__(self, eps=0.15, clip_max=0.5, clip_min=-0.5):
        super(FGSM, self).__init__(clip_max, clip_min)
        self.eps = eps

    def generate(self, model, x, y):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        ny = torch.unsqueeze(y, 0)
        nx.requires_grad_()
        out = model(nx)
        loss = F.cross_entropy(out, ny)
        loss.backward()
        x_adv = nx + self.eps * torch.sign(nx.grad.data)
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()

class BIM(Attacker):
    """
    Basic Iterative Method
    Alexey Kurakin, Ian J. Goodfellow, Samy Bengio.
    Adversarial Examples in the Physical World.
    arXiv, 2016
    """
    def __init__(self, eps=0.15, eps_iter=0.01, n_iter=50, clip_max=0.5, clip_min=-0.5):
        super(BIM, self).__init__(clip_max, clip_min)
        self.eps = eps
        self.eps_iter = eps_iter
        self.n_iter = n_iter

    def generate(self, model, x, y):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        ny = torch.unsqueeze(y, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        for i in range(self.n_iter):
            out = model(nx+eta)
            loss = F.cross_entropy(out, ny)
            loss.backward()

            eta += self.eps_iter * torch.sign(nx.grad.data)
            eta.clamp_(-self.eps, self.eps)
            nx.grad.data.zero_()

        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()






class PGD(Attacker):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
        
    """
    def __init__(self,  num_steps=40, step_size=0.01,eps=0.3, eps_norm='inf',step_norm='inf',clip_max=0.5, clip_min=-0.5, y_target=None):
        super(PGD, self).__init__(clip_max, clip_min)
        self.num_steps = num_steps
        self.step_size = step_size
        self.eps = eps
        self.eps_norm = eps_norm
        self.step_norm = step_norm
        self.y_target = y_target
        self.clip_max = clip_max
        self.clip_min = clip_min

    def generate(self, model, x, y):
        loss_fn = nn.CrossEntropyLoss()
        x = torch.unsqueeze(x, 0)
        y = torch.unsqueeze(y, 0)        
        r"""
        Overridden.
        """
        x_adv = x.clone().detach().requires_grad_(True).to(x.device)
        
        targeted = self.y_target is not None
        num_channels = x.shape[1]

        for i in range(self.num_steps):
            _x_adv = x_adv.clone().detach().requires_grad_(True)

            prediction = model(_x_adv)
            loss = loss_fn(prediction, self.y_target if targeted else y)
            loss.backward()

            with torch.no_grad():
                # Force the gradient step to be a fixed size in a certain norm
                if self.step_norm == 'inf':
                    gradients = _x_adv.grad.sign() * self.step_size
                else:
                    # Note .view() assumes batched image data as 4D tensor
                    gradients = _x_adv.grad * self.step_size / _x_adv.grad.view(_x_adv.shape[0], -1)\
                        .norm(self.step_norm, dim=-1)\
                        .view(-1, num_channels, 1, 1)

                if targeted:
                    # Targeted: Gradient descent with on the loss of the (incorrect) target label
                    # w.r.t. the image data
                    x_adv -= gradients
                else:
                    # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                    # the model parameters
                    x_adv += gradients

            # Project back into l_norm ball and correct range
            if self.eps_norm == 'inf':
                # Workaround as PyTorch doesn't have elementwise clip
                x_adv = torch.max(torch.min(x_adv, x + self.eps), x - self.eps)
            else:
                delta = x_adv - x

                # Assume x and x_adv are batched tensors where the first dimension is
                # a batch dimension
                mask = delta.view(delta.shape[0], -1).norm(norm, dim=1) <= self.eps

                scaling_factor = delta.view(delta.shape[0], -1).norm(norm, dim=1)
                scaling_factor[mask] = self.eps

                # .view() assumes batched images as a 4D Tensor
                delta *= self.eps / scaling_factor.view(-1, 1, 1, 1)

                x_adv = x + delta
                    
            x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()




class DeepFool(Attacker):
    """
    DeepFool
    Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Pascal Frossard
    DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks.
    CVPR, 2016
    """
    def __init__(self, max_iter=50, clip_max=0.5, clip_min=-0.5):
        super(DeepFool, self).__init__(clip_max, clip_min)
        self.max_iter = max_iter

    def generate(self, model, x, y):
        model.eval()
        nx = torch.unsqueeze(x, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)

        out = model(nx+eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = None

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = np.abs(fi.item()) / np.linalg.norm(wi.numpy().flatten())

                if value_i < value_l:
                    ri = value_i/np.linalg.norm(wi.numpy().flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()
            out = model(nx+eta)
            py = out.max(1)[1].item()
            i_iter += 1
        
        x_adv = nx + eta
        x_adv.clamp_(self.clip_min, self.clip_max)
        x_adv.squeeze_(0)
        
        return x_adv.detach()
