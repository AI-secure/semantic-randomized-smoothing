
import os
import sys
sys.path.append('.')
sys.path.append('..')

# evaluate a smoothed classifier on a dataset
import argparse
# import setGPU
from datasets import get_dataset, DATASETS, get_num_classes, get_normalize_layer
from semantic.core import SemanticSmooth
from math import ceil, sqrt
from time import time
import torch
import torchvision
import datetime
from tensorboardX import SummaryWriter
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
parser.add_argument("--th", type=float, default=0, help="pre-defined radius for true robust counting")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

if __name__ == "__main__":

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # init tensorboard writer
    writer = SummaryWriter(os.path.dirname(args.outfile))

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    if checkpoint["arch"] == 'resnet50' and args.dataset == "imagenet":
        try:
            base_classifier.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            # print(e)
            print('direct load failed, try alternative')
            try:
                base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
                base_classifier.load_state_dict(checkpoint['state_dict'])
            except Exception as e:
                print('direct load failed again... try alternative')
                base_classifier = torchvision.models.resnet50(pretrained=False).cuda()
                normalize_layer = get_normalize_layer('imagenet')
                base_classifier = torch.nn.Sequential(normalize_layer, base_classifier)
                base_classifier.load_state_dict(checkpoint['state_dict'])
    else:
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

    tot, tot_clean, tot_good = 0, 0, 0

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
                    assert wrongs[nearest] == 1
                    radius = disps[idx + nearest][0] ** 2 + disps[idx + nearest][1] ** 2
                    # print('radii', radii)
                    break
                except ValueError:
                    pass
                    # print('good')

                idx += this_batch_size
                if disps[idx-1][0] ** 2 + disps[idx-1][1] ** 2 > args.th ** 2:
                    radius = disps[idx-1][0] ** 2 + disps[idx-1][1] ** 2
                    break
                else:
                    print(disps[idx-1][0] ** 2 + disps[idx-1][1] ** 2, end='\r', flush=True)

        radius = sqrt(radius)
        correct = int(prediction == label)
        after_time = time()

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)
        print(i, time_elapsed, correct, radius)

        tot += 1
        tot_clean += correct
        tot_good += int(radius > args.th if correct > 0 else 0)
        writer.add_scalar('certify/clean_acc', tot_clean / tot, i)
        # writer.add_scalar('certify/robust_acc', tot_cert / tot, i)
        writer.add_scalar('certify/true_robust_acc', tot_good / tot, i)

    f.close()
