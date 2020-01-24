
import os

# evaluate a smoothed classifier on a dataset
import argparse
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from semantic.core import SemanticSmooth
from time import time
import torch
import torchvision
import datetime
from architectures import get_architecture
from semantic.transformers import gen_transformer
from semantic.transformers import BrightnessTransformer
from semantic.transforms import Translational, BlackTranslational

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument('transtype', type=str, help='type of semantic transformations',
                    choices=['translation', 'btranslation'])
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    if checkpoint["arch"] == 'resnet50' and args.dataset == "imagenet":
        try:
            base_classifier.load_state_dict(checkpoint['state_dict'])
        except:
            base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # prepare output file
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    if args.transtype == 'translation':
        print('translation')
        transform = Translational(dataset[0][0], 0.0)
    elif args.transtype == 'btranslation':
        print('black-padding translation')
        transform = BlackTranslational(dataset[0][0], 0.0)
    else:
        raise Exception('Unsupported transformation')

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # TODO
        # # certify the prediction of g around x
        # # x = x.cuda()
        # # prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        # after_time = time()
        # correct = int(prediction == label)
        #
        # time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        # print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
        #     i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        # print(i, time_elapsed, correct, radius)

    f.close()
