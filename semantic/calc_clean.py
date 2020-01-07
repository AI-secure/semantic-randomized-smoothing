
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
parser.add_argument("noise_sd", type=float, help="noise hyperparameter")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--verbstep", type=int, default=100, help="print for how many subslices")
args = parser.parse_args()

if __name__ == '__main__':

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

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

        print('working on #', i)
        before_time = time()
        cAHat = smoothed_classifier.guess_top(x.cuda(), args.N0, args.batch)

        clean_correct = (cAHat == label)

        tot, tot_good = tot + 1, tot_good + clean_correct
        print(f'{i} {clean_correct} RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)}')

    print(f'RACC = {tot_good}/{tot} = {float(tot_good) / float(tot)}')

