
import os
import sys
sys.path.append('.')
sys.path.append('..')

import math

# evaluate a smoothed classifier on a dataset
import argparse
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes, get_normalize_layer
from semantic.core import SemanticSmooth
from time import time
import random
import setproctitle
import torch
import torchvision
import datetime
from tensorboardX import SummaryWriter

from architectures import get_architecture
from semantic.transformers import RotationTransformer
from semantic.transformers import GaussianTransformer, RotationBrightnessNoiseTransformer, RotationBrightnessContrastNoiseTransformer
from semantic.transforms import visualize

EPS = 1e-6

parser = argparse.ArgumentParser(description='Strict rotation certify')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("noise_sd", type=float, help="noise hyperparameter")
parser.add_argument("--noise_b", type=float, default=0.0, help="noise hyperparameter for brightness shift dimension")
parser.add_argument("--noise_k", type=float, default=0.0, help="noise hyperparameter for brightness scaling dimension")
parser.add_argument("--l2_r", type=float, default=0.0, help="additional l2 magnitude to be tolerated")
parser.add_argument("aliasfile", type=str, help='output of alias data')
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--b", type=float, default=0.0, help="brightness shift requirement")
parser.add_argument("--k", type=float, default=0.0, help="brightness scaling requirement")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=500)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--slice", type=int, default=1000, help="number of angle slices")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--partial", type=float, default=180.0, help="certify +-partial degrees")
parser.add_argument("--verbstep", type=int, default=10, help="output frequency")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

if __name__ == '__main__':
    orig_alpha = args.alpha
    args.alpha /= (args.slice * (2.0 * args.partial) / 360.0 + 1)


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)

    if checkpoint["arch"] == 'resnet50' and args.dataset == "imagenet":
        try:
            base_classifier.load_state_dict(checkpoint['state_dict'])
        except:
            base_classifier = torchvision.models.resnet50(pretrained=False).cuda()

            # fix
            normalize_layer = get_normalize_layer('imagenet').cuda()
            base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)

    base_classifier.load_state_dict(checkpoint['state_dict'])

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    # init transformers
    rotationT = RotationTransformer(dataset[0][0])
    transformer = None
    # if abs(args.noise_b) < EPS and abs(args.noise_k) < EPS:
    #     transformer = GaussianTransformer(args.noise_sd)
    if abs(args.noise_k) < EPS:
        transformer = RotationBrightnessNoiseTransformer(args.noise_sd, args.noise_b, dataset[0][0], 0.)
        transformer.set_brightness_shift(args.b)
        transformer.rotation_adder.mask = transformer.rotation_adder.mask.cuda()
    else:
        transformer = RotationBrightnessContrastNoiseTransformer(args.noise_sd, args.noise_b, args.noise_k, dataset[0][0], 0.)
        transformer.set_brightness_shift(args.b)
        transformer.set_brightness_scale(1.0 - args.k, 1.0 + args.k)
        transformer.rotation_adder.mask = transformer.rotation_adder.mask.cuda()

    # init alias analysis
    alias_dic = dict()
    f = open(args.aliasfile, 'r')
    for line in f.readlines()[1:]:
        try:
            no, v = line.split('\t')
        except:
            # sometimes we have the last column for time recording
            no, v, _ = line.split('\t')
        no, v = int(no), float(v)
        alias_dic[no] = v

    # modify outfile name to distinguish different parts
    if args.start != 0 or args.max != -1:
        args.outfile += f'_start_{args.start}_end_{args.max}'

    setproctitle.setproctitle(f'rotation_certify_{args.dataset}from{args.start}to{args.max}')

    # prepare output file
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # init tensorboard writer
    writer = SummaryWriter(os.path.dirname(args.outfile))

    # create the smooothed classifier g
    smoothed_classifier = SemanticSmooth(base_classifier, get_num_classes(args.dataset), transformer)

    tot, tot_clean, tot_good, tot_cert = 0, 0, 0, 0

    for i in range(len(dataset)):

        if i < args.start:
            continue

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i >= args.max >= 0:
            break

        (x, label) = dataset[i]
        if i not in alias_dic:
            continue

        margin = alias_dic[i]
        margin = (math.sqrt(margin) + args.l2_r) ** 2
        print('working on #', i, 'max aliasing:', alias_dic[i], '->', margin)

        before_time = time()
        cAHat = smoothed_classifier.predict(x.cuda(), args.N0, orig_alpha, args.batch)

        clean, cert, good = (cAHat == label), True, True
        gap = -1.0

        for j in range(args.slice):
            if min(360.0 * j / args.slice, 360.0 - 360.0 * (j + 1) / args.slice) >= args.partial:
                continue
            if j % args.verbstep == 0:
                print(f"> {j}/{args.slice} {str(datetime.timedelta(seconds=(time() - before_time)))}", end='\r', flush=True)
            now_x = rotationT.rotation_adder.raw_proc(x, 360.0 * j / args.slice).type_as(x).cuda()
            prediction, gap = smoothed_classifier.certify(now_x, args.N0, args.N, args.alpha, args.batch,
                                                          cAHat=cAHat, margin_sq=margin)
            if prediction != cAHat or gap < 0 or cAHat == smoothed_classifier.ABSTAIN:
                print(prediction)
                print(cAHat)
                print(gap)
                print(f'not robust @ slice #{j}')
                good = cert = False
                break
            elif prediction != label:
                # the prediction is robustly wrong:
                print(f'wrong @ slice #{j}')
                # make gap always smaller than 0 for wrong slice
                gap = - abs(gap) - 1.0
                good = False
                # robustly wrong is also skipped
                # now "cert" is not recorded anymore
                break
            # else it is good


        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, cAHat, gap, clean, time_elapsed), file=f, flush=True)

        tot, tot_clean, tot_cert, tot_good = tot + 1, tot_clean + int(clean), tot_cert + int(cert), tot_good + int(good)
        print(f'{i} {gap >= 0.0} '
              f'CleanACC = {tot_clean}/{tot} = {float(tot_clean) / float(tot)} '
              # f'CertAcc = {tot_cert}/{tot} = {float(tot_cert) / float(tot)} '
              f'RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)} '
              f'Time = {time_elapsed}')

        writer.add_scalar('certify/clean_acc', tot_clean / tot, i)
        # writer.add_scalar('certify/robust_acc', tot_cert / tot, i)
        writer.add_scalar('certify/true_robust_acc', tot_good / tot, i)

    print(f'CleanACC = {tot_clean}/{tot} = {float(tot_clean) / float(tot)} '
        # f'CertAcc = {tot_cert}/{tot} = {float(tot_cert) / float(tot)} '
        f'RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)}', file=f, flush=True)

    f.close()



