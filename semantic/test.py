
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
    rotationT = transforms.Rotation(canopy, rotation_angle=180.0)
    translationT = transforms.Translational(canopy, sigma=5.0)
    brightnessShiftT = transforms.BrightnessShift(sigma=0.1)
    brightnessScaleT = transforms.BrightnessScale(sigma=0.1)
    sizeScaleT = transforms.Resize(canopy, sl=0.5, sr=5.0)

    repeat_n = 10

    # transform image examples
    for loader, set_name in [(train_loader, 'train'), (test_loader, 'test')]:
        for i, (inputs, targets) in enumerate(loader):
            # only pick the first mini-batch
            for j in range(inputs.shape[0]):
                input_rep = inputs[j].repeat((repeat_n, 1, 1, 1))
                # outs = noiseT.batch_proc(input_rep)
                # for k in range(outs.shape[0]):
                #     transforms.visualize(outs[k], f'{args.outdir}/{args.dataset}/{set_name}/noise/{j}/{k}.bmp')
                # rots = rotationT.batch_proc(input_rep)
                # for k in range(outs.shape[0]):
                #     transforms.visualize(rots[k], f'{args.outdir}/{args.dataset}/{set_name}/rotation/{j}/{k}.bmp')
                # rotouts = rotationT.batch_proc(outs)
                # for k in range(outs.shape[0]):
                #     transforms.visualize(rots[k], f'{args.outdir}/{args.dataset}/{set_name}/rotation+noise/{j}/{k}.bmp')
                # transes = translationT.batch_proc(input_rep)
                # for k in range(transes.shape[0]):
                #     transforms.visualize(transes[k], f'{args.outdir}/{args.dataset}/{set_name}/translation/{j}/{k}.bmp')
                # shifts = brightnessShiftT.batch_proc(input_rep)
                # for k in range(shifts.shape[0]):
                #     transforms.visualize(shifts[k], f'{args.outdir}/{args.dataset}/{set_name}/brightnessShift/{j}/{k}.bmp')
                # scales = brightnessScaleT.batch_proc(input_rep)
                # for k in range(scales.shape[0]):
                #     transforms.visualize(scales[k], f'{args.outdir}/{args.dataset}/{set_name}/brightnessScale/{j}/{k}.bmp')
                sscales = torch.zeros_like(input_rep)
                for k in range(sscales.shape[0]):
                    sscales[k] = sizeScaleT.proc(input_rep[k], 5.0 / (k + 1))
                    transforms.visualize(sscales[k], f'{args.outdir}/{args.dataset}/{set_name}/resize/{j}/{k}.bmp')

            break

    # rotation aliasing measuring
    # for loader, set_name in [(train_loader, 'train'), (test_loader, 'test')]:
    #     for i, (inputs, targets) in enumerate(loader):
    #         # only pick the first mini-batch
    #         for j in range(inputs.shape[0]):
    #             for k in range(repeat_n):
    #                 angle = rotationT.gen_param()
    #                 orig = inputs[j]
    #                 orig_mask = rotationT.masking(orig)
    #                 rot = rotationT.raw_proc(orig, angle)
    #                 recover = rotationT.raw_proc(rot, -angle)
    #                 recover_mask = rotationT.masking(recover)
    #                 transforms.visualize(orig_mask, f'{args.outdir}/{args.dataset}/{set_name}/rotation-alias/{j}/{k}-orig.bmp')
    #                 transforms.visualize(recover_mask, f'{args.outdir}/{args.dataset}/{set_name}/rotation-alias/{j}/{k}-recover.bmp')
    #                 delta = torch.sum((orig_mask - recover_mask) * (orig_mask - recover_mask)).item()
    #                 print(set_name, j, k, 'angle:', angle, 'l2 delta:', delta)
    #         break

    # rotation small angle measuring
    # small_angle = 0.1
    # for loader, set_name in [(train_loader, 'train'), (test_loader, 'test')]:
    #     for i, (inputs, targets) in enumerate(loader):
    #         # only pick the first mini-batch
    #         for j in range(inputs.shape[0]):
    #             mean = 0.0
    #             maximum = 0.0
    #             for k in range(repeat_n):
    #                 angle = rotationT.gen_param()
    #                 anglep = angle + small_angle
    #                 orig = inputs[j]
    #                 rot = rotationT.raw_proc(orig, angle)
    #                 rotp = rotationT.raw_proc(orig, anglep)
    #                 rot = rotationT.masking(rot)
    #                 rotp = rotationT.masking(rotp)
    #                 delta = torch.sum((rot - rotp) * (rot - rotp)).item()
    #                 mean += delta
    #                 maximum = max(maximum, delta)
    #             mean /= repeat_n
    #             print(set_name, j, 'mean:', mean, 'maximum:', maximum)
    #         break

