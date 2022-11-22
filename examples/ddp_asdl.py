import os
import argparse
from collections import OrderedDict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import asdfghjkl as asdl

import wandb

OPTIM_SGD = 'sgd'
OPTIM_ADAM = 'adam'
OPTIM_KFAC = 'kfac'
OPTIM_SMW_NGD = 'smw_ngd'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'kron_psgd'
OPTIM_NEWTON = 'newton'
OPTIM_ABS_NEWTON = 'abs_newton'
OPTIM_KBFGS = 'kbfgs'
OPTIM_CURVE_BALL = 'curve_ball'
OPTIM_SENG = 'seng'
OPTIM_SHAMPOO = 'shampoo'

DEFAULT_MASTER_ADDR = '127.0.0.1'
DEFAULT_MASTER_PORT = '1234'


def init_dist_process_group(backend='nccl', is_high_priority=True):
    if os.environ.get('LOCAL_RANK', None) is not None:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_size = int(os.environ.get('LOCAL_SIZE', world_size))
    elif os.environ.get('SLURM_JOBID', None) is not None:
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        local_rank = 0
        world_rank = 0
        world_size = 1
        local_size = 1

    if world_size > 1:
        assert dist.is_available()
        master_addr = os.environ.get('MASTER_ADDR', DEFAULT_MASTER_ADDR)
        master_port = os.environ.get('MASTER_PORT', DEFAULT_MASTER_PORT)
        init_method = 'tcp://' + master_addr + ':' + master_port
        if backend == 'nccl' and is_high_priority:
            pg_options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        else:
            pg_options = None
        dist.init_process_group(backend,
                                init_method=init_method,
                                rank=world_rank,
                                world_size=world_size,
                                pg_options=pg_options)
        assert dist.get_rank() == world_rank
        assert dist.get_world_size() == world_size

    return local_rank, local_size, world_rank, world_size


def main():
    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(epoch)
        total_train_time += time.time() - start
        test(epoch)

    if world_rank == 0:
        print(f'total_train_time: {total_train_time:.2f}s')
        print(f'avg_epoch_time: {total_train_time / args.epochs:.2f}s')
        print(f'avg_step_time: {total_train_time / args.epochs / num_steps_per_epoch * 1000:.2f}ms')
        if args.wandb:
            wandb.run.summary['total_train_time'] = total_train_time
            wandb.run.summary['avg_epoch_time'] = total_train_time / args.epochs
            wandb.run.summary['avg_step_time'] = total_train_time / args.epochs / num_steps_per_epoch


def train(epoch):
    model.train()
    for batch_idx, (x, t) in enumerate(train_loader):
        x, t = x.cuda(), t.cuda()
        optimizer.zero_grad(set_to_none=True)

        # y = model(x)
        # loss = F.cross_entropy(y, t)
        # loss.backward()

        dummy_y = grad_maker.setup_model_call(model, x)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y, t)
        y, loss = grad_maker.forward_and_backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        if batch_idx % args.log_interval == 0 and world_rank == 0:
            if args.wandb:
                log = {'epoch': epoch,
                       'iteration': (epoch - 1) * num_steps_per_epoch + batch_idx + 1,
                       'train_loss': float(loss),
                       'learning_rate': optimizer.param_groups[0]['lr']}
                wandb.log(log)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / num_steps_per_epoch, float(loss)))

        scheduler.step()


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    if dist.is_initialized():
        packed = torch.tensor([correct, test_loss]).cuda()
        dist.reduce(packed, 0, dist.ReduceOp.SUM)
        correct, test_loss = packed.tolist()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    if world_rank == 0:
        if args.wandb:
            log = {'epoch': epoch,
                'iteration': epoch * num_steps_per_epoch,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy}
            wandb.log(log)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_accuracy))


if __name__ == '__main__':
    local_rank, local_size, world_rank, world_size = init_dist_process_group()

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optim', default=OPTIM_KFAC)
    parser.add_argument('--damping', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb', action='store_true', default=False)

    args = parser.parse_args()
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    torch.cuda.reset_accumulated_memory_stats()

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size, 'drop_last': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    common_kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
    train_kwargs.update(common_kwargs)
    test_kwargs.update(common_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)

    if dist.is_initialized():
        train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
        train_kwargs.update({'sampler': train_sampler})
        test_sampler = DistributedSampler(dataset=test_set, shuffle=True)
        test_kwargs.update({'sampler': test_sampler})

    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)
    num_steps_per_epoch = len(train_loader)

    
    hidden_dim = args.hidden_dim
    model = nn.Sequential(OrderedDict([
        ('flatten', nn.Flatten()),
        ('fc1', nn.Linear(784, hidden_dim)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_dim, hidden_dim)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(hidden_dim, 10)),
    ])).cuda()
    

    #model = models.resnet18().cuda()

    if args.optim == OPTIM_ADAM:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.optim == OPTIM_KFAC:
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            damping=args.damping)
        grad_maker = asdl.KfacGradientMaker(model, config)
    elif args.optim == OPTIM_SMW_NGD:
        config = asdl.SmwEmpNaturalGradientConfig(data_size=args.batch_size,
                                                  damping=args.damping)
        grad_maker = asdl.SmwEmpNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_FULL_PSGD:
        grad_maker = asdl.PsgdGradientMaker(model)
    elif args.optim == OPTIM_KRON_PSGD:
        grad_maker = asdl.KronPsgdGradientMaker(model)
    elif args.optim == OPTIM_NEWTON:
        config = asdl.NewtonGradientConfig(damping=args.damping)
        grad_maker = asdl.NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_ABS_NEWTON:
        config = asdl.NewtonGradientConfig(damping=args.damping, absolute=True)
        grad_maker = asdl.NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_KBFGS:
        config = asdl.KronBfgsGradientConfig(data_size=args.batch_size,
                                             damping=args.damping)
        grad_maker = asdl.KronBfgsGradientMaker(model, config)
    elif args.optim == OPTIM_CURVE_BALL:
        config = asdl.CurveBallGradientConfig(damping=args.damping)
        grad_maker = asdl.CurveBallGradientMaker(model, config)
    elif args.optim == OPTIM_SENG:
        config = asdl.SengGradientConfig(data_size=args.batch_size,
                                         damping=args.damping)
        grad_maker = asdl.SengGradientMaker(model, config)
    elif args.optim == OPTIM_SHAMPOO:
        config = asdl.ShampooGradientConfig(damping=args.damping)
        grad_maker = asdl.ShampooGradientMaker(model, config)
    else:
        grad_maker = asdl.GradientMaker(model)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * num_steps_per_epoch)

    config = vars(args).copy()
    config.pop('wandb')
    if args.optim in [OPTIM_SGD, OPTIM_ADAM]:
        config.pop('damping')
    if args.wandb and world_rank == 0:
        wandb.init(config=config,
                   entity=os.environ.get('WANDB_ENTITY', None),
                   project=os.environ.get('WANDB_PROJECT', None),
                   )

    if world_rank == 0:
        print('=====================')
        for key, value in config.items():
            print(f'{key}: {value}')
        print('=====================')

    torch.cuda.synchronize()
    try:
        main()
        max_memory = torch.cuda.max_memory_allocated()
    except RuntimeError as err:
        if 'CUDA out of memory' in str(err):
            print(err)
            max_memory = -1  # OOM
        else:
            raise RuntimeError(err)

    print(f'cuda_max_memory: {max_memory/float(1<<30):.2f}GB')
    if args.wandb and world_rank == 0:
        wandb.run.summary['cuda_max_memory'] = max_memory
