import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .core import extend
from .operations import OP_BATCH_GRADS

__all__ = ['data_loader_gradient', 'batch_gradient', 'save_batch_gradient', 'jacobian']


def data_loader_gradient(
    model,
    loss_fn,
    data_loader,
    is_distributed=False,
    all_reduce=False,
    is_master=True,
    data_average=True
):
    # NOTE: loss_fn is supposed be defined with reduction='sum'

    # accumulate gradient for data_loader
    device = next(model.parameters()).device
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        total_loss += loss.item()

    # take average of accumulated gradient
    if data_average:
        data_size = len(data_loader.dataset)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(data_size)
        total_loss /= data_size

    # reduce gradient and total_loss
    if is_distributed:
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        # pack
        packed_tensor = torch.cat([parameters_to_vector(grads),
                                   torch.tensor(total_loss, device=device)])
        # reduce
        if all_reduce:
            dist.all_reduce(packed_tensor)
        else:
            dist.reduce(packed_tensor, dst=0)
        # unpack
        if is_master or all_reduce:
            total_loss = packed_tensor[-1].item()
            packed_tensor = packed_tensor[:-1]
            vector_to_parameters(
                packed_tensor.div_(dist.get_world_size()), grads
            )

        dist.barrier()

    return total_loss


def batch_gradient(model, closure, return_outputs=False):
    """
    Calculates gradients of parameters of the model for each batch.
    Args:
        model: torch.nn.Module instance to calculate gradient for.
        closure: A function which does backpropagation.
        return_outputs: If True, outputs of closure is returned as well.
    Returns:
        Batch gradients of the shape (n, p) where n is the batch size and p is the number of parameters.
        If return_outputs is True, outputs of closure is also returned.
    Example::
        >>> model = torch.nn.Linear(100, 10)
        >>> x = torch.randn(32, 100)
        >>> y = torch.tensor([0]*32, dtype=torch.long)
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        >>> def closure():
        ...     outputs = model(x)
        ...     loss = loss_fn(outputs, y)
        ...     loss.backward()
        ...     return outputs
        ...
        >>> batch_grads = asdl.batch_gradient(model, closure, return_outputs=True)
        >>> for grad in batch_grads:
        ...     print(grad.shape)
        ...
        torch.Size([32, 1010])
        torch.Size([32, 10])
    """
    with extend(model, OP_BATCH_GRADS) as cxt:
        outputs = closure()
        grads = []
        for module in model.modules():
            g = cxt.batch_grads(module, flatten=True)
            if g is not None:
                grads.append(g)
        grads = torch.cat(grads, dim=1)  # (n, p)
    if return_outputs:
        return grads, outputs
    else:
        return grads


def save_batch_gradient(model, closure, return_outputs=False):
    """
    Calculate batch gradient of the model's parameters and save it to 'batch_grad' attribute of each parameter.
    Args:
        model: torch.nn.Module instance to calculate gradient for.
        closure: A function which does backpropagation.
        return_outputs: If True, outputs of closure is returned as well.
    Returns:
        Outputs of closure if return_outputs is True.
    Example::
        >>> model = torch.nn.Linear(100, 10)
        >>> x = torch.randn(32, 100)
        >>> y = torch.tensor([0]*32, dtype=torch.long)
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        >>> def closure():
        ...     outputs = model(x)
        ...     loss = loss_fn(outputs, y)
        ...     loss.backward()
        ...     return outputs
        ...
        >>> save_batch_gradient(model, closure)
        >>> for param in model.parameters():
        ...     print(param.batch_grad.shape)
        ...
        torch.Size([32, 10, 100])
        torch.Size([32, 10])
    """
    with extend(model, OP_BATCH_GRADS) as cxt:
        outputs = closure()
        for module in model.modules():
            grads = cxt.batch_grads(module)
            if grads is not None:
                for key, value in grads.items():
                    param = getattr(module, key)
                    if hasattr(param, 'batch_grad'):
                        param.batch_grad += value
                    else:
                        setattr(param, 'batch_grad', value)
    if return_outputs:
        return outputs

    
def jacobian(model, x):
    f = model(x)
    assert f.ndim == 2  # (n, c)
    n, c = f.shape
    rst = []
    for i in range(c):
        with extend(model, OP_BATCH_GRADS):
            model.zero_grad()
            loss = f[:, i].sum()
            loss.backward()
        grads = [p.batch_grads for p in model.parameters() if p.requires_grad]
        grads = torch.hstack([g.view(n, -1) for g in grads])  # (n, p)
        rst.append(grads)
    return torch.stack(rst).transpose(0, 1)  # (n, c, p)
