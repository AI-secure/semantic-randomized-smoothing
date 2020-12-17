
import os
import sys
sys.path.append('.')
sys.path.append('..')

# evaluate a smoothed classifier on a dataset
import argparse
# import setGPU
from tensorboardX import SummaryWriter
from datasets import get_dataset, DATASETS, get_num_classes, get_normalize_layer
from semantic.core import SemanticSmooth
from time import time
import torch
import torchvision
import datetime
from architectures import get_architecture
from semantic.transformers import gen_transformer

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("noise_sd", type=float, help="noise hyperparameter")
parser.add_argument('transtype', type=str, help='type of semantic transformations',
                    choices=['rotation-noise', 'noise', 'rotation', 'translation', 'brightness', 'contrast', 'gaussian', 'expgaussian'])
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument('--noise_k', default=0.0, type=float,
                    help="standard deviation of brightness scaling")
parser.add_argument('--noise_b', default=0.0, type=float,
                    help="standard deviation of brightness shift")
parser.add_argument("--bright_scale", type=float, default=0.0,
                    help="for brightness transformation, the scale interval is 1.0 +- bright_scale")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument("--th", type=float, default=0, help="pre-defined radius for true robust counting")
args = parser.parse_args()

if __name__ == "__main__":

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

    # prepare output file
    if not os.path.exists(os.path.dirname(args.outfile)):
        os.makedirs(os.path.dirname(args.outfile))
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)

    transformer = gen_transformer(args, dataset[0][0])

    # special setting for brightness
    if args.transtype == 'brightness':
        transformer.set_brightness_scale(1.0 - args.bright_scale, 1.0 + args.bright_scale)
    if args.transtype == 'contrast':
        # binary search from 0.1 to 10.0
        transformer.set_contrast_scale(0.1, 10.0)

    # init tensorboard writer
    writer = SummaryWriter(os.path.dirname(args.outfile))

    # create the smooothed classifier g
    smoothed_classifier = SemanticSmooth(base_classifier, get_num_classes(args.dataset), transformer)

    tot_clean, tot_good, tot = 0, 0, 0

    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        # x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

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
