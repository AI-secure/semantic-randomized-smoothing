# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang, modified by Linyi Li.

import os
import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import torch
import torchvision
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS, get_normalize_layer
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from tensorboardX import SummaryWriter
from train_utils import AverageMeter, accuracy, init_logfile, log
from semantic.transformers import gen_transformer, AbstractTransformer

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('transtype', type=str, help='type of semantic transformations',
                    choices=['rotation-noise', 'noise', 'rotation', 'strict-rotation-noise', 'translation',
                             'brightness', 'resize', 'gaussian', 'btranslation', 'expgaussian', 'foldgaussian',
                             'rotation-brightness', 'rotation-brightness-contrast', 'resize-brightness'])
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")
parser.add_argument('--rotation_angle', help='constrain the rotation angle to +-rotation angle in degree',
                    type=float, default=180.0)
parser.add_argument('--noise_k', default=0.0, type=float,
                    help="standard deviation of brightness scaling")
parser.add_argument('--noise_b', default=0.0, type=float,
                    help="standard deviation of brightness shift")
parser.add_argument('--sl', default=1.0, type=float,
                    help="resize minimum ratio")
parser.add_argument('--sr', default=1.0, type=float,
                    help="resize maximum ratio")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrain', default=None, type=str)
##################### arguments for consistency training #####################
parser.add_argument('--num-noise-vec', default=1, type=int,
                    help="number of noise vectors. `m` in the paper.")
parser.add_argument('--lbd', default=20., type=float)
##################### arguments for tensorboard print #####################
parser.add_argument('--print_step', action="store_true")

args = parser.parse_args()


def kl_div(input, targets, reduction='batchmean'):
    return F.kl_div(F.log_softmax(input, dim=1), targets,
                    reduction=reduction)


def _cross_entropy(input, targets, reduction='mean'):
    targets_prob = F.softmax(targets, dim=1)
    xent = (-targets_prob * F.log_softmax(input, dim=1)).sum(1)
    if reduction == 'sum':
        return xent.sum()
    elif reduction == 'mean':
        return xent.mean()
    elif reduction == 'none':
        return xent
    else:
        raise NotImplementedError()


def _entropy(input, reduction='mean'):
    return _cross_entropy(input, input, reduction)

def main():
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)

    if args.pretrain is not None:
        if args.pretrain == 'torchvision':
            # load pretrain model from torchvision
            if args.dataset == 'imagenet' and args.arch == 'resnet50':
                model = torchvision.models.resnet50(True).cuda()

                # fix
                normalize_layer = get_normalize_layer('imagenet').cuda()
                model = torch.nn.Sequential(normalize_layer, model)


                print('loaded from torchvision for imagenet resnet50')
            else:
                raise Exception(f'Unsupported pretrain arg {args.pretrain}')
        else:
            # load the base classifier
            checkpoint = torch.load(args.pretrain)
            model.load_state_dict(checkpoint['state_dict'])
            print(f'loaded from {args.pretrain}')

    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
    writer = SummaryWriter(args.outdir)

    canopy = None
    for (inputs, targets) in train_loader:
        canopy = inputs[0]
        break
    transformer = gen_transformer(args, canopy)

    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    for epoch in range(args.epochs):
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, transformer, writer)
        test_loss, test_acc = test(test_loader, model, criterion, epoch, transformer, writer, args.print_freq)
        after = time.time()

        scheduler.step(epoch)

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

def _chunk_minibatch(batch, num_batches):
    X, y = batch
    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]



def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int,
          transformer: AbstractTransformer, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_reg = AverageMeter()
    confidence = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    for i, batch in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        mini_batches = _chunk_minibatch(batch, args.num_noise_vec)
        for inputs, targets in mini_batches:
            targets = targets.cuda()
            batch_size = inputs.size(0)

            noised_inputs = [transformer.process(inputs).cuda() for _ in range(args.num_noise_vec)]

            # augment inputs with noise
            inputs_c = torch.cat(noised_inputs, dim=0)
            targets_c = targets.repeat(args.num_noise_vec)

            logits = model(inputs_c)

            loss_xent = criterion(logits, targets_c)

            logits_chunk = torch.chunk(logits, args.num_noise_vec, dim=0)
            softmax = [F.softmax(logit, dim=1) for logit in logits_chunk]
            avg_softmax = sum(softmax) / args.num_noise_vec

            consistency = [kl_div(logit, avg_softmax, reduction='none').sum(1)
                           + _entropy(avg_softmax, reduction='none')
                           for logit in logits_chunk]
            consistency = sum(consistency) / args.num_noise_vec
            consistency = consistency.mean()

            loss = loss_xent + args.lbd * consistency

            avg_confidence = -F.nll_loss(avg_softmax, targets)

            acc1, acc5 = accuracy(logits, targets_c, topk=(1, 5))
            losses.update(loss_xent.item(), batch_size)
            losses_reg.update(consistency.item(), batch_size)
            confidence.update(avg_confidence.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

            if args.print_step:
                writer.add_scalar(f'epoch/{epoch}/loss/train', losses.avg, i)
                writer.add_scalar(f'epoch/{epoch}/loss/consistency', losses_reg.avg, i)
                writer.add_scalar(f'epoch/{epoch}/loss/avg_confidence', confidence.avg, i)
                writer.add_scalar(f'epoch/{epoch}/batch_time', batch_time.avg, i)
                writer.add_scalar(f'epoch/{epoch}/accuracy/train@1', top1.avg, i)
                writer.add_scalar(f'epoch/{epoch}/accuracy/train@5', top5.avg, i)

    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('loss/consistency', losses_reg.avg, epoch)
    writer.add_scalar('loss/avg_confidence', confidence.avg, epoch)
    writer.add_scalar('batch_time', batch_time.avg, epoch)
    writer.add_scalar('accuracy/train@1', top1.avg, epoch)
    writer.add_scalar('accuracy/train@5', top5.avg, epoch)

    return (losses.avg, top1.avg)


def test(loader, model, criterion, epoch, transformer: AbstractTransformer, writer=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs
            targets = targets.cuda()

            # augment inputs with noise
            inputs = transformer.process(inputs).cuda()

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    i, len(loader), batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1, top5=top5))

        if writer:
            writer.add_scalar('loss/test', losses.avg, epoch)
            writer.add_scalar('accuracy/test@1', top1.avg, epoch)
            writer.add_scalar('accuracy/test@5', top5.avg, epoch)

        return (losses.avg, top1.avg)

# def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, transformer: AbstractTransformer):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     end = time.time()
#
#     # switch to train mode
#     model.train()
#
#     for i, (inputs, targets) in enumerate(loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         inputs = inputs
#         targets = targets.cuda()
#
#         # augment inputs with noise
#         inputs = transformer.process(inputs).cuda()
#
#         # compute output
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#
#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#         losses.update(loss.item(), inputs.size(0))
#         top1.update(acc1.item(), inputs.size(0))
#         top5.update(acc5.item(), inputs.size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % args.print_freq == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                 epoch, i, len(loader), batch_time=batch_time,
#                 data_time=data_time, loss=losses, top1=top1, top5=top5))
#
#     return (losses.avg, top1.avg)
#
#
# def test(loader: DataLoader, model: torch.nn.Module, criterion, transformer: AbstractTransformer):
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     end = time.time()
#
#     # switch to eval mode
#     model.eval()
#
#     with torch.no_grad():
#         for i, (inputs, targets) in enumerate(loader):
#             # measure data loading time
#             data_time.update(time.time() - end)
#
#             inputs = inputs
#             targets = targets.cuda()
#
#             # augment inputs with noise
#             inputs = transformer.process(inputs).cuda()
#
#             # compute output
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#
#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#             losses.update(loss.item(), inputs.size(0))
#             top1.update(acc1.item(), inputs.size(0))
#             top5.update(acc5.item(), inputs.size(0))
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if i % args.print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     i, len(loader), batch_time=batch_time,
#                     data_time=data_time, loss=losses, top1=top1, top5=top5))
#
#         return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()
