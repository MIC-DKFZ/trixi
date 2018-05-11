import warnings
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def get_vanilla_image_gradient(model, inpt, err_fn, abs=False):
    if isinstance(model, torch.nn.Module):
        model.zero_grad()
    inpt = inpt.detach()
    inpt.requires_grad = True

    output = model(inpt)

    err = err_fn(output)
    err.backward()

    grad = inpt.grad + 0

    if isinstance(model, torch.nn.Module):
        model.zero_grad()

    if abs:
        grad = torch.abs(grad)
    return grad


@lru_cache(maxsize=32)
def get_guided_image_gradient(model: torch.nn.Module, inpt, err_fn, abs=False):
    def guided_relu_hook_function(module, grad_in, grad_out):
        if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU)):
            return (torch.clamp(grad_in[0], min=0.0),)

    model.zero_grad()

    ### Apply hooks
    hook_ids = []
    for mod in model.modules():
        hook_id = mod.register_backward_hook(guided_relu_hook_function)
        hook_ids.append(hook_id)

    inpt = inpt.detach()
    inpt.requires_grad = True

    output = model(inpt)

    err = err_fn(output)
    err.backward()

    grad = inpt.grad + 0

    model.zero_grad()
    for hooks in hook_ids:
        hooks.remove()

    if abs:
        grad = torch.abs(grad)
    return grad


@lru_cache(maxsize=32)
def get_smooth_image_gradient(model, inpt, err_fn, n_runs=20, eps=0.1, grad_type="vanilla"):
    grads = []
    for i in range(n_runs):
        inpt = inpt + torch.randn(inpt.size()).to(inpt.device) * eps
        if grad_type == "vanilla":
            single_grad = get_vanilla_image_gradient(model, inpt, err_fn)
        elif grad_type == "guided":
            single_grad = get_guided_image_gradient(model, inpt, err_fn)
        else:
            warnings.warn("This grad_type is not implemented yet")
            single_grad = torch.zeros_like(inpt)
        grads.append(torch.abs(single_grad))

    grad = torch.mean(torch.stack(grads), dim=0)
    return grad
