
import os
import math

# evaluate a smoothed classifier on a dataset
import argparse
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from semantic.core import StrictRotationSmooth
from time import time
import random
import torch
import datetime
from architectures import get_architecture
from semantic.transformers import ResizeTransformer
from semantic.transforms import visualize

parser = argparse.ArgumentParser(description='Resize certify')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sl", type=float, help="mininum resize ratio")
parser.add_argument("sr", type=float, help="maximum resize ratio")
parser.add_argument("noise_sd", type=float, help="noise hyperparameter")
parser.add_argument("aliasfile", type=str, help='output of alias data')
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--slice", type=int, default=1000, help="number of angle slices")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--verbstep", type=int, default=100, help="print for how many subslices")
args = parser.parse_args()

if __name__ == '__main__':

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    # init transformers
    resizeT = ResizeTransformer(dataset[0][0], args.sl, args.sr)

    # calculate params
    gbl_k = (1.0 / args.sl - 1.0 / args.sr) / (args.slice - 1)
    gbl_c = float(args.slice - 1) / (args.sr / args.sl - 1.0)

    # init alias analysis
    alias_dic = dict()
    f = open(args.aliasfile, 'r')
    for line in f.readlines()[1:]:
        no, v = line.split('\t')
        no, v = int(no), float(v)
        alias_dic[no] = v

    # prepare output file
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # create the smooothed classifier g
    smoothed_classifier = StrictRotationSmooth(base_classifier, get_num_classes(args.dataset), args.noise_sd)

    tot, tot_good = 0, 0

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        if i not in alias_dic:
            continue

        print('working on #', i, 'max aliasing:', alias_dic[i])
        margin = alias_dic[i]

        before_time = time()
        cAHat = smoothed_classifier.guess_top(x.cuda(), args.N0, args.batch)

        good = False
        gap = -1.0
        if cAHat == label:
            good = True
            for j in range(args.slice):
                s = min(max(1.0 / (gbl_k * (j + gbl_c)), args.sl), args.sr)
                if j % args.verbstep == 0:
                    print(f"> {j}/{args.slice} ratio: {s}")
                now_x = resizeT.resizer.proc(x, s).cuda()
                prediction, gap = smoothed_classifier.certify(now_x, cAHat, args.N, args.alpha, args.batch, margin)
                if prediction != label or gap < 0:
                    print(f'wrong @ slice #{j}')
                    good = False
                    break

        after_time = time()
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, cAHat, gap, good, time_elapsed), file=f, flush=True)

        tot, tot_good = tot + 1, tot_good + good
        print(f'{i} {good} RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)}')

    f.close()



