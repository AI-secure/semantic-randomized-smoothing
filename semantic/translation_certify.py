
import os

# evaluate a smoothed classifier on a dataset
import argparse
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from semantic.core import SemanticSmooth
from math import ceil, sqrt
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
parser.add_argument("--batch", type=int, default=1000, help="batch size")
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
    base_classifier.eval()

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

    # generate displacements
    _, h, w = dataset[0][0].shape
    print(h, w)
    disps = [(i, j) for j in range(-w, w + 1) for i in range(-h, h + 1)]
    disps = sorted(disps, key=(lambda a: a[0]**2 + a[1]**2))
    num = len(disps)

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()

        radius = disps[-1][0] ** 2 + disps[-1][1] ** 2

        # print(i)

        with torch.no_grad():
            idx = 0
            for _ in range(ceil(num / args.batch)):
                this_batch_size = min(args.batch, num - idx)

                batch = torch.zeros((this_batch_size,) + x.shape)
                for j in range(this_batch_size):
                    # batch[i] = x
                    batch[j] = transform.proc(x, disps[idx + j][0], disps[idx + j][1])


                batch = batch.cuda()
                predictions = base_classifier(batch).argmax(1)
                if idx == 0:
                    prediction = predictions[0].item()

                wrongs = (predictions != label).tolist()
                try:
                    nearest = wrongs.index(1)
                    radius = disps[idx + nearest][0] ** 2 + disps[idx + nearest][1] ** 2
                    assert wrongs[nearest] == 1
                    # print('radii', radii)
                    break
                except ValueError:
                    pass
                    # print('good')

                idx += this_batch_size

        radius = sqrt(radius)
        correct = int(prediction == label)
        after_time = time()

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        print(i, time_elapsed, correct, radius)

    f.close()
