import os
import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import time
import datetime
import math
from numpy import arange

import torch
import torchvision
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Uniform

from datasets import get_dataset, DATASETS, get_normalize_layer, get_dataset_shape
from architectures import ARCHITECTURES, get_architecture
from train_utils import AverageMeter, accuracy, init_logfile, log
import semantic.transforms as T


parser = argparse.ArgumentParser(description='PyTorch Grid Search Noive Certificaton')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('outfile', type=str, help='folder to save model and training log)')
parser.add_argument('transtype', type=str, help='type of semantic transformations',
                    choices=['brightness', 'contrast', 'brightness-contrast', 'rotation', 'resize', 'rotation-brightness'])
parser.add_argument('--param1', default=None, type=float,
                    help='attack param1')
parser.add_argument('--param2', default=None, type=float,
                    help='attack param2')
parser.add_argument('--batch', default=400, type=int,
                    help='batch size')
parser.add_argument('--tries1', default=10000, type=int,
                    help='tries for param1')
parser.add_argument('--tries2', default=None, type=int,
                    help='tries for param2')
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--verbstep", type=int, default=100, help="output frequency")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()


if __name__ == '__main__':
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)


    model = get_architecture(checkpoint["arch"], args.dataset)
    if checkpoint["arch"] == 'resnet50' and args.dataset == "imagenet":
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model = torchvision.models.resnet50(pretrained=False).cuda()

    # if args.dataset == 'imagenet':
    #     # directly load from torchvision instead of original architecture API
    #     model = torchvision.models.resnet50(False).cuda()
    #     normalize_layer = get_normalize_layer('imagenet').cuda()
    #     model = torch.nn.Sequential(normalize_layer, model)
    #     print('loaded from torchvision for imagenet resnet50')
    # else:

    # model = get_architecture(checkpoint["arch"], args.dataset)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    cells = 1
    for x in get_dataset_shape(args.dataset):
        cells *= x


    # init transformers
    tinst = None
    tfunc = None
    tinst2 = None
    tfunc2 = None
    if args.transtype == 'brightness':
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
    elif args.transtype == 'rotation-brightness':
        tinst = T.Rotation(dataset[0][0], 0.0)
        tfunc = T.Rotation.proc
        tinst2 = T.BrightnessShift(0.0)
        tfunc2 = T.BrightnessShift.proc

    # rand generator
    if args.transtype == 'contrast':
        param1l, param1r = math.log(1.0 - args.param1), math.log(1.0 + args.param1)
    elif args.transtype == 'resize':
        param1l, param1r = 1.0 - args.param1, 1.0 + args.param1
    else:
        param1l, param1r = -args.param1, +args.param1

    if args.transtype == 'brightness-contrast':
        param2l, param2r = math.log(1.0 - args.param2), math.log(1.0 + args.param2)
    elif args.transtype == 'rotation-brightness':
        param2l, param2r = -args.param2, +args.param2
    else:
        param2l, param2r = None, None

    tot = tot_benign = tot_robust = tot_cert = 0

    if args.dataset == 'mnist':
        core = model[1]

        global_L = 1.
        for layer in core:
            if isinstance(layer, torch.nn.Conv2d):
                norm = torch.norm(layer.weight.data.reshape((layer.out_channels, -1)), p='fro')
                print('conv2d', norm)
                global_L *= norm
            elif isinstance(layer, torch.nn.Linear):
                norm = torch.norm(layer.weight.data, p='fro')
                print('linear', norm)
                global_L *= norm
        print(global_L)
        global_L = global_L.item()


    for i in range(len(dataset)):

        if i < args.start:
            continue

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i >= args.max >= 0:
            break

        print('working on #', i)
        stime = time.time()
        (x, y) = dataset[i]

        clean_x = x.cuda().unsqueeze(0)
        logits = model(clean_x)[0]
        pred = logits.argmax()
        mask = torch.zeros_like(logits)
        mask[pred] = - (max(logits) - min(logits))
        runner_up = (logits + mask).argmax()

        if pred != y:
            pass
        else:
            tot_benign += 1
            robust = True


            if args.transtype == 'brightness':
                require_gap = math.sqrt(cells) * (param1r - param1l) / args.tries1
            elif args.transtype == 'contrast':
                require_gap = (math.exp(param1r) - math.exp(param1r - (param1r - param1l) / args.tries1)) * torch.norm(clean_x.reshape(-1), p=2)
            print('require gap', require_gap)

            # var_x = Variable(clean_x.data, requires_grad=True).contiguous()
            # opt = torch.optim.Adam([var_x], lr=1e-3)
            # opt.zero_grad()
            # model(var_x)[0][pred].backward()
            # grad1_len = torch.norm(var_x.grad.data).item()
            # opt.zero_grad()
            # model(var_x)[0][runner_up].backward()
            # grad2_len = torch.norm(var_x.grad.data).item()
            # print(logits[pred] - logits[runner_up], grad1_len + grad2_len, (logits[pred] - logits[runner_up]) / (grad1_len + grad2_len))

            # local_lip_lb = 0.
            closest_gap = 1e+20

            for j in range(0, args.tries1, args.batch):
                now_batch = min(args.batch, args.tries1 - j)
                xp = torch.zeros((now_batch, ) + tuple(x.shape))

                param_sample1 = arange(param1l + (param1r - param1l) / args.tries1 * j,
                                       param1l + (param1r - param1l) / args.tries1 * (j + now_batch),
                                       (param1r - param1l) / args.tries1)
                # if param2l is not None:
                #     param_sample1 = arange(param2l + (param2r - param2l) / args.tries2 * j,
                #                            param2l + (param2r - param2l) / args.tries2 * (j + now_batch),
                #                            (param2r - param2l) / args.tries2)

                for k in range(now_batch):
                    xp[k] = tfunc(tinst, x, param_sample1[k])
                # if param2l is not None:
                #     for k in range(now_batch):
                #         xp[k] = tfunc2(tinst2, xp[k], param_sample2[k])

                # var_xp = Variable(xp.data.cuda(), requires_grad=True).contiguous()
                # opt = torch.optim.Adam([var_xp], lr=1e-3)
                # opt.zero_grad()
                # (model(var_xp)[:,pred]).sum(dim=0).backward()
                # # torch.autograd.backward(model(var_xp)[:,pred], grad_tensors=None)
                # grad1_len = torch.norm(var_xp.grad.data.reshape(now_batch, -1), p=2, dim=1).max().item()
                # opt.zero_grad()
                # (model(var_xp)[:,runner_up]).sum(dim=0).backward()
                # # torch.autograd.backward(model(var_xp)[:,runner_up], grad_tensors=None)
                # # model(var_xp)[:,runner_up].backward()
                # grad2_len = torch.norm(var_xp.grad.data.reshape(now_batch, -1), p=2, dim=1).max().item()
                # local_lip_lb = max(local_lip_lb, grad1_len + grad2_len)
                # # print(grad1_len + grad2_len)

                xp = xp.contiguous().cuda()
                logits = model(xp)

                closest_gap = min(closest_gap, min(logits[:,pred] - logits[:,runner_up]).item())

                preds = logits.argmax(1)
                if sum((preds != y).type(torch.long)) > 0:
                    robust = False
                    break

            print('now_gap', closest_gap, global_L, closest_gap / global_L)
            if closest_gap / global_L > require_gap:
                tot_cert += 1

            tot_robust += int(robust)


        tot += 1
        ttime = time.time()
        print(f'#{i} clean acc={tot_benign / tot} robust acc={tot_robust / tot} robust cert={tot_cert / tot} time={ttime-stime} s')


    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'w')
    f.write(f'clean {tot_benign / tot},{tot_benign} attack_rob {tot_robust / tot},{tot_robust} cert_rob {tot_cert / tot} {tot}\n')
    f.write(f'param1 {args.param1} param2 {args.param2} tries1 {args.tries1} tries2 {args.tries2}\n')
    f.close()
