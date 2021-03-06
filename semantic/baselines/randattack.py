import os
import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import time
import datetime
import math

import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Uniform, Beta

from datasets import get_dataset, DATASETS, get_normalize_layer
from architectures import ARCHITECTURES, get_architecture
from train_utils import AverageMeter, accuracy, init_logfile, log
import semantic.transforms as T

EPS = 1e-6


parser = argparse.ArgumentParser(description='PyTorch Grid Search Noive Certificaton')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('outfile', type=str, help='file to output attack result')
parser.add_argument('transtype', type=str, help='type of semantic transformations',
                    choices=['gaussian', 'translation',  'brightness', 'contrast', 'brightness-contrast', 'rotation', 'resize', 'resize-brightness', 'rotation-brightness', 'rotation-brightness-contrast'])
parser.add_argument('--param1', default=None, type=float,
                    help='attack param1')
parser.add_argument('--param2', default=None, type=float,
                    help='attack param2')
parser.add_argument('--param3', default=None, type=float,
                    help='attack param2')
parser.add_argument('--batch', default=400, type=int,
                    help='batch size')
parser.add_argument('--tries', default=10000, type=int,
                    help='attack param2')
parser.add_argument('--l2', default=None, type=float, help='additional l2-bounded perturbations')
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--verbstep", type=int, default=100, help="output frequency")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()



def fgsm(model, X, y, epsilon, niters=100, alpha=0.02):
    out = model(X)
    ce = torch.nn.CrossEntropyLoss()(out, y)
    err = (out.data.max(1)[1] != y.data).float().sum() / X.size(0)

    X_fgsm = Variable(X.data, requires_grad=True).contiguous()

    opt = torch.optim.Adam([X_fgsm], lr=1e-3)
    opt.zero_grad()
    loss = torch.nn.CrossEntropyLoss()(model(X_fgsm), y)
    loss.backward()
    grad = X_fgsm.grad.data

    unit_grad = F.normalize(grad, p=2, dim=list(range(1, grad.dim())))
    delta = unit_grad * epsilon
    x_fgsm = X + delta

    return x_fgsm

    # for i in range(niters):
    #     opt = torch.optim.Adam([X_pgd], lr=1e-3)
    #     opt.zero_grad()
    #     loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y)
    #     loss.backward()
    #     eta = alpha * X_pgd.grad.data.sign()
    #     X_pgd = Variable(X_pgd.data + eta, requires_grad=True).cuda()
    #
    #     # adjust to be within [-epsilon, epsilon]
    #     eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    #     X_pgd = Variable(X.data + eta, requires_grad=True).cuda()
    #
    # err_output = model(X_pgd)
    # ce = torch.nn.CrossEntropyLoss()(err_output, y)
    #
    # # err_acc = (model(X_pgd).data.max(1)[1] == y.data).float().sum() * 100. / X.size(0)
    #
    # err_acc = (model(X_pgd).data.max(1)[1] == y.data).float().sum() / X.shape[0]
    # return X_pgd, err_output, ce.item(), err_acc.item()


if __name__ == '__main__':
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    if args.dataset == 'imagenet':
        # directly load from torchvision instead of original architecture API
        model = torchvision.models.resnet50(False).cuda()
        normalize_layer = get_normalize_layer('imagenet').cuda()
        model = torch.nn.Sequential(normalize_layer, model)
        print('loaded from torchvision for imagenet resnet50')
    else:
        model = get_architecture(checkpoint["arch"], args.dataset)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)


    # init transformers
    tinst = None
    tfunc = None
    tinst2 = None
    tfunc2 = None
    tinst3 = None
    tfunc3 = None
    if args.transtype == 'gaussian':
        tinst = T.Gaussian(0.0)
        tfunc = T.Gaussian.proc
    elif args.transtype == 'translation':
        tinst = T.Translational(dataset[0][0], 0.0)
        tfunc = T.Translational.proc
    elif args.transtype == 'brightness':
        tinst = T.BrightnessShift(0.0)
        tfunc = T.BrightnessShift.proc
    elif args.transtype == 'contrast':
        # note: contrast is in exponential scale
        tinst = T.BrightnessScale(0.0)
        tfunc = T.BrightnessScale.proc
    elif args.transtype == 'brightness-contrast':
        tinst = T.BrightnessShift(0.0)
        tfunc = T.BrightnessShift.proc
        tinst2 = T.BrightnessScale(0.0)
        tfunc2 = T.BrightnessScale.proc
    elif args.transtype == 'rotation':
        tinst = T.Rotation(dataset[0][0], 0.0)
        # tfunc = T.Rotation.proc
        tfunc = T.Rotation.raw_proc
    elif args.transtype == 'resize':
        # note: resize is in original scale
        tinst = T.Resize(dataset[0][0], 1.0, 1.0)
        tfunc = T.Resize.proc
    elif args.transtype == 'resize-brightness':
        # note: resize is in original scale
        tinst = T.Resize(dataset[0][0], 1.0, 1.0)
        tfunc = T.Resize.proc
        tinst2 = T.BrightnessShift(0.0)
        tfunc2 = T.BrightnessShift.proc
    elif args.transtype == 'rotation-brightness':
        tinst = T.Rotation(dataset[0][0], 0.0)
        # tfunc = T.Rotation.proc
        tfunc = T.Rotation.raw_proc
        tinst2 = T.BrightnessShift(0.0)
        tfunc2 = T.BrightnessShift.proc
    elif args.transtype == 'rotation-brightness-contrast':
        tinst = T.Rotation(dataset[0][0], 0.0)
        # tfunc = T.Rotation.proc
        tfunc = T.Rotation.raw_proc
        tinst2 = T.BrightnessShift(0.0)
        tfunc2 = T.BrightnessShift.proc
        tinst3 = T.BrightnessScale(0.0)
        tfunc3 = T.BrightnessScale.proc

    # rand generator

    if args.transtype == 'gaussian':
        param1l, param1r = 0.0, args.param1
    elif args.transtype == 'contrast':
        param1l, param1r = math.log(1.0 - args.param1), math.log(1.0 + args.param1)
    elif args.transtype == 'resize' or args.transtype == 'resize-brightness':
        param1l, param1r = 1.0 - args.param1, 1.0 + args.param1
    else:
        param1l, param1r = -args.param1, +args.param1

    param3l, param3r = None, None
    if args.transtype == 'brightness-contrast':
        param2l, param2r = math.log(1.0 - args.param2), math.log(1.0 + args.param2)
    elif args.transtype == 'rotation-brightness' or args.transtype == 'resize-brightness':
        param2l, param2r = -args.param2, +args.param2
    elif args.transtype == 'rotation-brightness-contrast':
        param2l, param2r = -args.param2, +args.param2
        param3l, param3r = -args.param3, +args.param3
    else:
        param2l, param2r = None, None

    if args.transtype == 'translation':
        candidates = torch.tensor(list(set([(x, y) for x in range(int(args.param1) + 1) for y in range(int(args.param1) + 1)
                       if float(x * x + y * y) <= args.param1 * args.param1] +
                      [(x, -y) for x in range(int(args.param1) + 1) for y in range(int(args.param1) + 1)
                       if float(x * x + y * y) <= args.param1 * args.param1] +
                      [(-x, y) for x in range(int(args.param1) + 1) for y in range(int(args.param1) + 1)
                       if float(x * x + y * y) <= args.param1 * args.param1] +
                      [(-x, -y) for x in range(int(args.param1) + 1) for y in range(int(args.param1) + 1)
                       if float(x * x + y * y) <= args.param1 * args.param1]
                      )))
        c_len = candidates.shape[0]
        param1l, param1r = 0.0, c_len

    # m1 = Uniform(param1l, param1r)
    m1 = Beta(0.5, 0.5)
    if param2l is not None:
        m2 = Beta(0.5, 0.5)
        # m2 = Uniform(param2l, param2r)
    if param3l is not None:
        m3 = Beta(0.5, 0.5)
        # m3 = Uniform(param3l, param3r)

    tot = tot_benign = tot_robust = 0

    for i in range(len(dataset)):

        if i < args.start:
            continue

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i >= args.max >= 0:
            break

        print('working on #', i)
        (x, y) = dataset[i]

        clean_x = x.cuda().unsqueeze(0)
        pred = model(clean_x).argmax()
        if pred != y:
            pass
        else:
            tot_benign += 1
            robust = True

            for j in range(0, args.tries, args.batch):
                now_batch = min(args.batch, args.tries - j)
                xp = torch.zeros((now_batch, ) + tuple(x.shape))

                if args.transtype == 'translation':
                    param_sample1 = candidates[(m1.sample((now_batch,)) * (param1r - param1l) + param1l).type(torch.long)]
                    for k in range(now_batch):
                        xp[k] = tfunc(tinst, x, param_sample1[k][0].item(), param_sample1[k][1].item())
                else:
                    param_sample1 = m1.sample((now_batch,)) * (param1r - param1l) + param1l
                    if param2l is not None:
                        param_sample2 = m2.sample((now_batch,)) * (param2r - param2l) + param2l
                    if param3l is not None:
                        param_sample3 = m3.sample((now_batch,)) * (param3r - param3l) + param3l

                    for k in range(now_batch):
                        xp[k] = tfunc(tinst, x, param_sample1[k])
                    if param2l is not None:
                        for k in range(now_batch):
                            xp[k] = tfunc2(tinst2, xp[k], param_sample2[k])
                    if param3l is not None:
                        for k in range(now_batch):
                            xp[k] = tfunc3(tinst3, xp[k], param_sample3[k])

                xp = xp.contiguous().cuda()
                xp_old = xp

                if args.l2 is not None and args.l2 > EPS:
                    xp = fgsm(model, xp, torch.tensor([y], dtype=torch.long).expand(now_batch).cuda(), args.l2)

                # print(torch.norm((xp_old - xp).reshape(xp.size()[0], -1), dim=1))

                preds = model(xp).argmax(1)
                if sum((preds != y).type(torch.long)) > 0:
                    robust = False
                    break

            tot_robust += int(robust)


        tot += 1
        print(f'#{i} clean acc={tot_benign / tot} robust acc={tot_robust / tot}')


    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'a')
    f.write(f'clean {tot_benign / tot},{tot_benign} robust={tot_robust / tot},{tot_robust} {tot}\n')
    f.write(f'param1 {args.param1} param2 {args.param2} tries {args.tries}\n')
    f.close()
