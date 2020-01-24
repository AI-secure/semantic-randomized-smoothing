
import os
import math

# evaluate a smoothed classifier on a dataset
import argparse
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from semantic.core import StrictRotationSmooth
from time import time
import random
import setproctitle
import torch
import datetime
from architectures import get_architecture
from semantic.transformers import RotationTransformer, NoiseTransformer
from semantic.transforms import visualize

parser = argparse.ArgumentParser(description='Strict rotation certify')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("noise_sd", type=float, help="noise hyperparameter")
parser.add_argument("aliasfile", type=str, help='output of alias data')
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--start", type=int, default=0, help="start before skipping how many examples")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--slice", type=int, default=1000, help="number of angle slices")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--partial", type=float, default=180.0, help="certify +-partial degrees")
parser.add_argument("--verbstep", type=int, default=100, help="certify +-partial degrees")
args = parser.parse_args()

if __name__ == '__main__':

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    # init transformers
    rotationT = RotationTransformer(dataset[0][0])
    noiseT = NoiseTransformer(args.noise_sd)

    # init alias analysis
    alias_dic = dict()
    f = open(args.aliasfile, 'r')
    for line in f.readlines()[1:]:
        no, v = line.split('\t')
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

    # create the smooothed classifier g
    smoothed_classifier = StrictRotationSmooth(base_classifier, get_num_classes(args.dataset), args.noise_sd)

    tot, tot_good = 0, 0

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

        print('working on #', i, 'max aliasing:', alias_dic[i])
        margin = alias_dic[i]

        before_time = time()
        cAHat = smoothed_classifier.guess_top(x.cuda(), args.N0, args.batch)

        # good = False
        clean = False
        gap = -1.0
        if cAHat == label:
            # good = True
            clean = True
            for j in range(args.slice):
                if min(360.0 * j / args.slice, 360.0 - 360.0 * (j + 1) / args.slice) >= args.partial:
                    continue
                if j % args.verbstep == 0:
                    print(f"> {j}/{args.slice} {str(datetime.timedelta(seconds=(time() - before_time)))}")
                now_x = rotationT.rotation_adder.raw_proc(x, 360.0 * j / args.slice).cuda()
                prediction, gap = smoothed_classifier.certify(now_x, cAHat, args.N, args.alpha, args.batch, margin)
                if prediction != label or gap < 0:
                    print(f'wrong @ slice #{j}')
                    # make gap always smaller than 0 for wrong slice
                    gap = - abs(gap) - 1.0
                    # good = False
                    break

        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, cAHat, gap, clean, time_elapsed), file=f, flush=True)

        tot, tot_good = tot + 1, tot_good + (gap >= 0.0)
        print(f'{i} {gap >= 0.0} RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)}')

    f.close()



