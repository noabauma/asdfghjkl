import math
from pathlib import Path
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
from torch.utils.data import RandomSampler, DataLoader
import torchvision
from torchvision import transforms

import asdfghjkl as asdl


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='kfac')

    parser.add_argument('--data-path',
                        default='/project/g34/imagenet',
                        type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--val-batch_size', default=2048, type=int)
    parser.add_argument('--label-smoothing', default=0.1, type=float)

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--warmup-factor', type=float, default=0.125)
    parser.add_argument('--warmup-epochs', type=float, default=5)
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[15, 25, 30])

    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.00005)
    parser.add_argument('--cov-update-freq', type=int, default=10)
    parser.add_argument('--inv-update-freq', type=int, default=100)
    parser.add_argument('--ema-decay', type=float, default=0.05)
    parser.add_argument('--damping', type=float, default=0.001)
    parser.add_argument('--kl-clip', type=float, default=0.001)

    return parser.parse_args()


def to_vector(x):
    return nn.utils.parameters_to_vector(x)


class Dataset(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.num_workers = 4
        self.pin_memory = True
        self.train_sampler = RandomSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       sampler=self.train_sampler,
                                       num_workers=12,
                                       pin_memory=True)
        self.val_sampler = RandomSampler(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.val_batch_size,
                                     sampler=self.val_sampler,
                                     num_workers=12,
                                     pin_memory=True)
        self.sampler = None
        self.loader = None

    def train(self):
        self.sampler = self.train_sampler
        self.loader = self.train_loader

    def eval(self):
        self.sampler = self.val_sampler
        self.loader = self.val_loader


class IMAGENET(Dataset):
    def __init__(self, args):
        self.num_classes = 1000
        self.train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'train'),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
        self.val_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'val'),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
        super().__init__(args)



class Metric(object):
    def __init__(self, device):
        self._n = torch.tensor([0.0]).to(device)
        self._loss = torch.tensor([0.0]).to(device)
        self._corrects = torch.tensor([0.0]).to(device)

    def update(self, n, loss, outputs, targets):
        with torch.inference_mode():
            self._n += n
            self._loss += loss * n
            _, preds = torch.max(outputs, 1)
            self._corrects += torch.sum(preds == targets)

    @property
    def loss(self):
        return (self._loss / self._n).item()

    @property
    def accuracy(self):
        return (self._corrects / self._n).item()

    def __str__(self):
        return f'Loss: {self.loss:.4f}, Acc: {self.accuracy:.4f}'


def train(epoch, dataset, model, criterion, opt, grad_maker, args):
    dataset.train()
    model.train()

    lr = opt.param_groups[0]['lr']
    metric = Metric('cuda')
    for i, (inputs, targets) in enumerate(dataset.loader):
        inputs = inputs.cuda()
        targets = targets.cuda()
        opt.zero_grad(set_to_none=True)

        # y = model(x)
        # loss = F.cross_entropy(y, t)
        # loss.backward()

        dummy_y = grad_maker.setup_model_call(model, inputs)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y, targets)
        y, loss = grad_maker.forward_and_backward()

        opt.step()

        metric.update(inputs.shape[0], loss, outputs, targets)

        if i % 100 == 0:
            print(f'Epoch {epoch} {i}/{len(dataset.loader)} Train {metric}')

    print(f'Epoch {epoch} Train {metric} LR: {lr}')


def test(epoch, dataset, model, criterion, args):
    dataset.eval()
    model.eval()

    metric = Metric(args.device)

    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(dataset.loader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            metric.update(inputs.shape[0], loss, outputs, targets)

    print(f'Epoch {epoch} Test {metric}')


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # ========== DATA ==========
    print("data load", flush=True)
    dataset = IMAGENET(args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    print("data load done", flush=True)

    # ========== MODEL ==========
    model = torchvision.models.resnet50(num_classes=dataset.num_classes)
    model.cuda()

    # ========== OPTIMIZER ==========
    opt = torch.optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # =========== GRAD MAKER =========
    config = asdl.NaturalGradientConfig(data_size=args.batch_size, damping=args.damping)
    grad_maker = asdl.KfacGradientMaker(model, config)

    # ========== LEARNING RATE SCHEDULER ==========
    if args.warmup_epochs > 0:
        lr_scheduler = SequentialLR(opt, [
            LinearLR(opt, args.warmup_factor, total_iters=args.warmup_epochs),
            MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1),
        ], [args.warmup_epochs])
    else:
        lr_scheduler = MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1)

    # ========== TRAINING ==========
    for e in range(args.epochs):

        train(e, dataset, model, criterion, opt, grad_maker, args)
        torch.cuda.empty_cache()
        test(e, dataset, model, criterion, args)
        torch.cuda.empty_cache()
        lr_scheduler.step()
