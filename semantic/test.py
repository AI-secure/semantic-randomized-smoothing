
import argparse
from torch.utils.data import DataLoader
import torch

from datasets import DATASETS, get_dataset
import semantic.transforms as transforms

parser = argparse.ArgumentParser(description='Transformation debugger')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('outdir', type=str)
parser.add_argument('--batch', type=int, default=8)
args = parser.parse_args()

if __name__ == '__main__':
    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch)

    # gen canopy
    for (inputs, targets) in train_loader:
        canopy = inputs[0]
        break

    # init transformers
    noiseT = transforms.Noise(sigma=0.5)
    rotationT = transforms.Rotation(canopy)

    repeat_n = 10

    for loader, set_name in [(train_loader, 'train'), (test_loader, 'test')]:
        for i, (inputs, targets) in enumerate(loader):
            # only pick the first mini-batch
            for j in range(inputs.shape[0]):
                input_rep = inputs[j].repeat((repeat_n, 1, 1, 1))
                outs = noiseT.batch_proc(input_rep)
                for k in range(outs.shape[0]):
                    transforms.visualize(outs[k], f'{args.outdir}/{args.dataset}/{set_name}/noise/{j}/{k}.bmp')
                rots = rotationT.batch_proc(input_rep)
                for k in range(outs.shape[0]):
                    transforms.visualize(rots[k], f'{args.outdir}/{args.dataset}/{set_name}/rotation/{j}/{k}.bmp')
                rotouts = rotationT.batch_proc(outs)
                for k in range(outs.shape[0]):
                    transforms.visualize(rots[k], f'{args.outdir}/{args.dataset}/{set_name}/rotation+noise/{j}/{k}.bmp')
            break


    # rotation aliasing measuring

    for loader, set_name in [(train_loader, 'train'), (test_loader, 'test')]:
        for i, (inputs, targets) in enumerate(loader):
            # only pick the first mini-batch
            for j in range(inputs.shape[0]):
                for k in range(repeat_n):
                    angle = rotationT.gen_param()
                    orig = inputs[j]
                    orig_mask = rotationT.masking(orig)
                    rot = rotationT.raw_proc(orig, angle)
                    recover = rotationT.raw_proc(rot, -angle)
                    recover_mask = rotationT.masking(recover)
                    transforms.visualize(orig_mask, f'{args.outdir}/{args.dataset}/{set_name}/rotation-alias/{j}/{k}-orig.bmp')
                    transforms.visualize(recover_mask, f'{args.outdir}/{args.dataset}/{set_name}/rotation-alias/{j}/{k}-recover.bmp')
                    delta = torch.sum((orig_mask - recover_mask) * (orig_mask - recover_mask)).item()
                    print(set_name, j, k, 'angle:', angle, 'l2 delta:', delta)
            break

