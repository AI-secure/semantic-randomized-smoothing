"""
    The random attack for our trained smoothed classifiers
"""

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
from torch.distributions import Uniform

from datasets import get_dataset, DATASETS, get_normalize_layer, get_num_classes
from architectures import ARCHITECTURES, get_architecture
from train_utils import AverageMeter, accuracy, init_logfile, log
from semantic.core import SemanticSmooth
import semantic.transformers as T
import semantic.transforms as TR

EPS = 1e-6


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to load pytorch model of base classifier")
parser.add_argument("noise_sd", type=float, help="noise hyperparameter")
parser.add_argument('transtype', type=str, help='type of semantic transformations',
                    choices=['gaussian', 'translation', 'brightness', 'brightness-contrast', 'rotation', 'scaling',
                             'rotation-brightness', 'scaling-brightness',
                             'rotation-brightness-l2', 'scaling-brightness-l2'])
parser.add_argument('--noise_k', default=0.0, type=float,
                    help="standard deviation of brightness scaling")
parser.add_argument('--noise_b', default=0.0, type=float,
                    help="standard deviation of brightness shift")
parser.add_argument('--blur_alpha', default=None, type=float, help='range of Guassian blurring: 0 <= alpha <= blur_alpha')
parser.add_argument('--displacement', default=None, type=float, help='range of translation: 0 <= |Delta| <= displacement')
parser.add_argument('--b', default=None, type=float, help='brightness change region: -b <= 0 <= b')
parser.add_argument('--k', default=None, type=float, help='contrast change ratio: 1-k <= 0 <= 1+k')
parser.add_argument('--r', default=None, type=float, help='rotation angle: -r <= 0 <= +r')
parser.add_argument('--s', default=None, type=float, help='scaling change ratio: 1-s <= 0 <= 1+s')
parser.add_argument('--l2_r', default=None, type=float, help='additional l2 noise added to the transformed image')
parser.add_argument('--tries', default=100, type=int, help='number of samples for parameter sampling')
parser.add_argument('--N0', default=1000, type=int, help='number of samples for estimating the most probable class')
parser.add_argument('--batch', default=400, type=int, help='number of samples on GPU')
parser.add_argument('--p', default=0.001, type=float, help='confidence level for binom test')
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument('--outfolder', default='new_data/smooth_attack', type=str,
                    help='folder to output attack result for smoothed classifier')
parser.add_argument('--outfile', default=None, type=str,
                    help='concrete file name to output attack result. '
                         'By default it is blank so that the file name is auto-generated')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()


def gen_inference_transformer(args, canopy):
    if args.transtype == 'gaussian':
        print(f'gaussian with exponential noise from 0 to {args.noise_sd}')
        return T.ExpGaussianTransformer(args.noise_sd)
    elif args.transtype == 'translation':
        print(f'translation with noise {args.noise_sd}')
        return T.TranslationTransformer(args.noise_sd, canopy)
    elif args.transtype == 'brightness':
        print(f'Use brightness sigma = {args.noise_b}')
        return T.BrightnessTransformer(0.0, args.noise_b)
    elif args.transtype == 'brightness-contrast':
        print(f'Use brightness sigma = {args.noise_b} and contrast sigma = {args.noise_k}')
        return T.BrightnessTransformer(args.noise_k, args.noise_b)
    elif args.transtype in ['rotation', 'rotation-brightness', 'rotation-brightness-l2']:
        if args.transtype == 'rotation':
            print(f'Use rotation = noise sigma {args.noise_sd} w/ masking')
            args.noise_b = 0.
            nowT = T.RotationBrightnessNoiseTransformer(args.noise_sd, args.noise_b, canopy, 0.)
        else:
            print(f'Use rotation + brightness = noise sigma {args.noise_sd}, brightness sigma {args.noise_b} w/masking')
            nowT = T.RotationBrightnessNoiseTransformer(args.noise_sd, args.noise_b, canopy, 0.)
        nowT.set_round(1)
        # nowT.rotation_adder.mask = nowT.rotation_adder.mask.cuda()
        return nowT
    elif args.transtype in ['scaling', 'scaling-brightness', 'scaling-brightness-l2']:
        if args.transtype == 'scaling':
            print(f'Use scaling = noise sigma {args.noise_sd}')
            args.noise_b = 0.
            nowT = T.ResizeBrightnessNoiseTransformer(args.noise_sd, args.noise_b, canopy, 1.0, 1.0)
        else:
            print(f'Use scaling + brightness = noise sigma {args.noise_sd}, brightness sigma {args.noise_b}')
            nowT = T.ResizeBrightnessNoiseTransformer(args.noise_sd, args.noise_b, canopy, 1.0, 1.0)
        return nowT

def gen_transform_and_params(args, canopy):

    # init transformers
    tinst1 = None
    tfunc1 = None
    tinst2 = None
    tfunc2 = None
    if args.transtype == 'gaussian':
        tinst1 = TR.Gaussian(0.0)
        tfunc1 = TR.Gaussian.proc
    elif args.transtype == 'translation':
        tinst1 = TR.Translational(canopy, 0.0)
        tfunc1 = TR.Translational.proc
    elif args.transtype == 'brightness':
        tinst1 = TR.BrightnessShift(0.0)
        tfunc1 = TR.BrightnessShift.proc
    elif args.transtype == 'brightness-contrast':
        # note: contrast is in exponential scale
        tinst1 = TR.BrightnessShift(0.0)
        tfunc1 = TR.BrightnessShift.proc
        tinst2 = TR.BrightnessScale(0.0)
        tfunc2 = TR.BrightnessScale.proc
    elif args.transtype == 'rotation':
        tinst1 = TR.Rotation(canopy, 0.0)
        tfunc1 = TR.Rotation.raw_proc
    elif args.transtype == 'scaling':
        # note: resize is in original scale
        tinst1 = TR.Resize(canopy, 1.0, 1.0)
        tfunc1 = TR.Resize.proc
    elif args.transtype == 'rotation-brightness' or args.transtype == 'rotation-brightness-l2':
        tinst1 = TR.Rotation(canopy, 0.0)
        tfunc1 = TR.Rotation.raw_proc
        tinst2 = TR.BrightnessShift(0.0)
        tfunc2 = TR.BrightnessShift.proc
    elif args.transtype == 'scaling-brightness' or args.transtype == 'scaling-brightness-l2':
        # note: resize is in original scale
        tinst1 = TR.Resize(canopy, 1.0, 1.0)
        tfunc1 = TR.Resize.proc
        tinst2 = TR.BrightnessShift(0.0)
        tfunc2 = TR.BrightnessShift.proc

    # random generator
    param1l, param1r, param2l, param2r, candidates = None, None, None, None, None
    if args.transtype == 'gaussian':
        param1l, param1r = 0.0, args.blur_alpha
    elif args.transtype == 'translation':
        candidates = torch.tensor(list(set([(x, y) for x in range(int(args.displacement) + 1) for y in range(int(args.displacement) + 1)
                       if float(x * x + y * y) <= args.displacement * args.displacement] +
                      [(x, -y) for x in range(int(args.displacement) + 1) for y in range(int(args.displacement) + 1)
                       if float(x * x + y * y) <= args.displacement * args.displacement] +
                      [(-x, y) for x in range(int(args.displacement) + 1) for y in range(int(args.displacement) + 1)
                       if float(x * x + y * y) <= args.displacement * args.displacement] +
                      [(-x, -y) for x in range(int(args.displacement) + 1) for y in range(int(args.displacement) + 1)
                       if float(x * x + y * y) <= args.displacement * args.displacement]
                      )))
        c_len = candidates.shape[0]
        param1l, param1r = 0.0, c_len
    elif args.transtype == 'brightness':
        param1l, param1r = -args.b, +args.b
    elif args.transtype == 'brightness-contrast':
        param1l, param1r = -args.b, +args.b
        param2l, param2r = math.log(1.0 - args.k), math.log(1.0 + args.k)
    elif args.transtype == 'rotation':
        param1l, param1r = -args.r, +args.r
    elif args.transtype == 'scaling':
        param1l, param1r = 1.0 - args.s, 1.0 + args.s
    elif args.transtype == 'rotation-brightness' or args.transtype == 'rotation-brightness-l2':
        param1l, param1r = -args.r, +args.r
        param2l, param2r = -args.b, +args.b
    elif args.transtype == 'scaling-brightness' or args.transtype == 'scaling-brightness-l2':
        param1l, param1r = 1.0 - args.s, 1.0 + args.s
        param2l, param2r = -args.b, +args.b
    else:
        raise Exception(f'Unknown transtype: {args.transtype}')

    print(f"""
    param1: [{param1l}, {param1r}]
    param2: [{param2l}, {param2r}]
""")
    return tinst1, tfunc1, tinst2, tfunc2, param1l, param1r, param2l, param2r, candidates



def main(args):
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    print('arch:', checkpoint['arch'])
    if checkpoint["arch"] == 'resnet50' and args.dataset == "imagenet":
        try:
            base_classifier.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print('direct load failed, try alternative')
            try:
                base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
                base_classifier.load_state_dict(checkpoint['state_dict'])
                # fix
                # normalize_layer = get_normalize_layer('imagenet').cuda()
                # base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
            except Exception as e:
                print('alternative failed again, try alternative 2')
                base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
                # base_classifier.load_state_dict(checkpoint['state_dict'])
                normalize_layer = get_normalize_layer('imagenet').cuda()
                base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
                base_classifier.load_state_dict(checkpoint['state_dict'])
    else:
        base_classifier.load_state_dict(checkpoint['state_dict'])


    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    # generate transformer
    transformer = gen_inference_transformer(args, dataset[0][0])
    smoothed_classifier = SemanticSmooth(base_classifier, get_num_classes(args.dataset), transformer)

    # generate image-level transform and params
    tinst1, tfunc1, tinst2, tfunc2, param1l, param1r, param2l, param2r, candidates = gen_transform_and_params(args, dataset[0][0])

    # init random number generator
    m1 = Uniform(param1l, param1r)
    if param2l is not None:
        m2 = Uniform(param2l, param2r)

    # init metrics
    tot = tot_benign = tot_robust = 0

    # [main] attack section
    for i in range(len(dataset)):

        # only certify every args.skip examples
        if i % args.skip != 0:
            continue

        print('working on #', i)
        (x, y) = dataset[i]

        # clean_x = x.cuda().unsqueeze(0)
        pred = smoothed_classifier.predict(x, args.N0, args.p, args.batch)
        if pred != y:
            pass
        else:
            tot_benign += 1
            robust = True

            for j in range(0, args.tries):
                xp = None

                if args.transtype == 'translation':
                    param_sample1 = candidates[int(m1.sample().item())]
                    xp = tfunc1(tinst1, x, param_sample1[0].item(), param_sample1[1].item())
                else:
                    param_sample1 = m1.sample().item()
                    if param2l is not None:
                        param_sample2 = m2.sample().item()

                    xp = tfunc1(tinst1, x, param_sample1)
                    if param2l is not None:
                        xp = tfunc2(tinst2, xp, param_sample2)

                # xp = xp.contiguous().cuda()
                # xp_old = xp

                # if args.l2 is not None and args.l2 > EPS:
                #     xp = fgsm(model, xp, torch.tensor([y], dtype=torch.long).expand(now_batch).cuda(), args.l2)

                # print(torch.norm((xp_old - xp).reshape(xp.size()[0], -1), dim=1))

                xp = xp.type_as(x)

                if args.transtype in ['rotation-brightness-l2', 'scaling-brightness-l2']:
                    # compute the gradient by soft label and empirical mean
                    smoothed_classifier.base_classifier.eval()
                    grad = torch.zeros((args.N0, xp.shape[0], xp.shape[1], xp.shape[2]))

                    n = 0
                    while n < args.N0:
                        now_batch = min(args.batch, args.N0 - n)
                        batch = xp.repeat((now_batch, 1, 1, 1))
                        batch_noised = smoothed_classifier.transformer.process(batch).cuda()
                        batch_noised = Variable(batch_noised.data, requires_grad=True).contiguous()
                        opt = torch.optim.Adam([batch_noised], lr=1e-3)
                        opt.zero_grad()
                        loss = torch.nn.CrossEntropyLoss()(smoothed_classifier.base_classifier(batch_noised),
                                                           torch.tensor([y], dtype=torch.long).expand(now_batch).cuda())
                        loss.backward()
                        grad[n: n+now_batch, :, :, :] = batch_noised.grad.data

                        n += now_batch

                    grad = torch.mean(grad, dim=0)
                    unit_grad = F.normalize(grad, p=2, dim=list(range(grad.dim())))
                    delta = unit_grad * args.l2_r

                    # print(xp)

                    xp = xp + delta

                    # print(xp + delta)
                    # print(delta)
                    # print(torch.norm(delta.reshape(-1)))

                pred = smoothed_classifier.predict(xp, args.N0, args.p, args.batch)
                if (pred != y):
                    robust = False
                    break

                print(f"> {j}/{args.tries}", end='\r', flush=True)

            tot_robust += int(robust)

        tot += 1
        print(f'#{i} clean acc={tot_benign / tot} robust acc={tot_robust / tot}')

    if args.outfile is None:
        param_str = ''
        if args.transtype != 'translation':
            param_str = f'{param1r}'
            if param2r is not None:
                param_str += f'_{param2r}'
        else:
            param_str = f'{args.displacement}'
        args.outfile = args.transtype + '/' + args.dataset + '/' + param_str + '/' + 'result.txt'
    out_full_path = os.path.join(args.outfolder, args.outfile)
    print('output result to ' + out_full_path)


    if not os.path.exists(os.path.dirname(out_full_path)):
        os.makedirs(os.path.dirname(out_full_path))

    f = open(out_full_path, 'w')
    f.write(f'clean {tot_benign / tot},{tot_benign} robust={tot_robust / tot},{tot_robust} tot={tot}\n')
    f.close()

    print('done')


if __name__ == '__main__':
    main(args)


