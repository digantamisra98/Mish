import argparse
import os
import random
import shutil
import time
from datetime import datetime

import wandb
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from resnet import ResidualNet
from PIL import ImageFile
from ptflops import get_model_complexity_info

ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--depth', default=50, type=int, metavar='D',
                        help='model depth')
    parser.add_argument('--ngpu', default=4, type=int, metavar='G',
                        help='number of gpus to use')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--log-freq', '-l', default=500, type=int,
                        metavar='L', help='log frequency (default: 500)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--seed", type=int, default=1234, metavar='BS',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--prefix", type=str, required=True, metavar='PFX',
                        help='prefix for logging & checkpoint saving')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluation only')
    parser.add_argument('--dryrun', required=False, type=bool)
    parser.add_argument("--entity", type=str, required=True, metavar='PFX',
                        help='Name of the WandB Entity')
    parser.add_argument("--project", type=str, required=True, metavar='PFX',
                        help='Name of the WandB project')
    args = parser.parse_args()
    return args


def init_wandb(entity, project, model):
    wandb.init(entity=entity, project=project)
    wandb.config.update(args)
    wandb.watch(model)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(arch, depth):
    assert arch == "resnet"
    model = ResidualNet('ImageNet', depth, 1000)
    print(model)
    return model


def get_model_stats(model, device, verbose=True):
    with torch.cuda.device(device):
        macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False,
                                                 print_per_layer_stat=False)
        if verbose:
            print('{:<30}  {:<8}'.format('Computational complexity: ', int(macs)))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs, params


def get_loss_optim(model, device, lr, momentum, weight_decay):
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    return criterion, optimizer


def get_model_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
        return start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(path))
        exit()


def get_dataloader(root_dir, is_train, batch_size, workers):
    dir_name = "train" if is_train else "val"
    data_dir = os.path.join(root_dir, dir_name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transformations = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ] if is_train else [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
    dataset = datasets.ImageFolder(data_dir, transforms.Compose(transformations))
    shuffle = True if is_train else False
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True)
    return loader


best_prec1 = 0
args = parse_args()
torch.backends.cudnn.benchmark = True


def main():
    global args, best_prec1
    print("args", args)

    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model("resnet", int(args.depth))
    macs, params = get_model_stats(model, device)
    model = model.to(device)

    criterion, optimizer = get_loss_optim(model, device, args.lr, args.momentum, args.weight_decay)

    model = torch.nn.DataParallel(model, device_ids=list(range(int(args.ngpu))))

    if args.resume:
        start_epoch = get_model_checkpoint(args.resume, model, optimizer)
        args.start_epoch = start_epoch

    if args.dryrun:
        for epoch in range(args.start_epoch, 100):
            print(epoch)
            x = torch.randn(args.batch_size, 3, 224, 224).to(device)
            model.zero_grad()
            y = model(x)
            y.mean().backward()
        exit()

    val_loader = get_dataloader(args.data, False, args.batch_size, args.workers)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    train_loader = get_dataloader(args.data, True, args.batch_size, args.workers)

    init_wandb(args.entity, args.project, model)
    wandb.config.update({"Parameters": params, "FLOPs": macs})
    print(f"Parameters: {params}, FLOPs: {macs}")
    print(args)
    print(model)
    for epoch in range(args.start_epoch, args.epochs):
        start_time = datetime.now()

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.prefix)

        end_time = datetime.now()
        delta = (end_time - start_time).total_seconds() // 60.
        wandb.log({'epoch': epoch, "best_prec1": best_prec1, "Time (min.)": delta})
        print(f"Epoch {epoch} Prec {prec1:.3f} Best {best_prec1:.3f} Time {delta} m")


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, top1=top1,
                                                                  top5=top5))
        if i % args.log_freq == 0:
            wandb.log(
                {"Batch": epoch * len(train_loader) + i, "Batch Training time (ms)": batch_time.val*10, "Batch Data time (ms)": data_time.val*10,
                 "Batch Training loss": losses.val, "Batch Training Top-1 accuracy": top1.val,
                 "Batch Training Top-5 accuracy": top5.val})


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            target = target.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    # log stats to wandb
    wandb.log({
        'epoch': epoch,
        'Top-1 accuracy': top1.avg,
        'Top-5 accuracy': top5.avg,
        'loss': losses.avg,
    })

    return top1.avg


def save_checkpoint(state, is_best, prefix):
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar' % prefix)
        wandb.save(filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.arch == 'mobilenet':
        lr = args.lr * (0.98 ** epoch)
    elif args.arch == 'resnet':
        lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    wandb.log({'lr': lr, 'epoch': epoch})


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
