
import argparse
from torch.utils.data import DataLoader
import torch
import math
import random

from datasets import DATASETS, get_dataset
import semantic.transforms as transforms

dir = 'visualize'

datasets = ['cifar10', 'mnist']
nums = [0, 20, 40, 60, 80, 100]


if __name__ == '__main__':

    for dataset in datasets:

        ds = get_dataset(dataset, 'test')

        canopy = ds[0][0]

        # init transformers
        noiseT = transforms.Noise(sigma=0.5)
        rotationT = transforms.Rotation(canopy, rotation_angle=180.0)
        translationT = transforms.Translational(canopy, sigma=5.0)
        blackTranslationT = transforms.BlackTranslational(canopy, sigma=5.0)
        brightnessShiftT = transforms.BrightnessShift(sigma=0.1)
        brightnessScaleT = transforms.BrightnessScale(sigma=0.1)
        sizeScaleT = transforms.Resize(canopy, sl=0.5, sr=5.0)
        gaussianT = transforms.Gaussian(sigma=5.0)

        for num in nums:
            print(dataset, num)
            transforms.visualize(ds[num][0], f'visualize/{dataset}/{num}.png')
            # rotation
            angles = [-10, 30, 70]
            for angle in angles:
                transforms.visualize(rotationT.masking(rotationT.raw_proc(ds[num][0], angle)), f'visualize/{dataset}/{num}_rot_{angle}.png')
            transforms.visualize(rotationT.masking(rotationT.raw_proc(rotationT.raw_proc(ds[num][0], angles[0]), angles[1])), f'visualize/{dataset}/{num}_rot_add20.png')
            transforms.visualize(rotationT.masking(rotationT.raw_proc(ds[num][0], angles[0] + angles[1])), f'visualize/{dataset}/{num}_rot_direct20.png')
            # translation
            ts = [(-3,-5), (-4, 2), (3, -5), (4, 2), (0, -5), (0, 6), (-2, 0), (-3, 0)]
            for (tx, ty) in ts:
                transforms.visualize(translationT.proc(ds[num][0], tx, ty), f'visualize/{dataset}/{num}_trans_{tx}_{ty}.png')
                transforms.visualize(blackTranslationT.proc(ds[num][0], tx, ty), f'visualize/{dataset}/{num}_blacktrans_{tx}_{ty}.png')
            # brightness
            bs = [0.2, 0.1, -0.1]
            for b in bs:
                transforms.visualize(brightnessShiftT.proc(ds[num][0], b), f'visualize/{dataset}/{num}_brightness_{b}.png')
            # contrast
            cs = [1.5, 0.9, 0.6]
            for c in cs:
                transforms.visualize(brightnessScaleT.proc(ds[num][0], math.log(c)), f'visualize/{dataset}/{num}_contrast_{c}.png')
            # gaussian
            gs = [5.0, 3.0, 1.0]
            for g in gs:
                transforms.visualize(gaussianT.proc(ds[num][0], g), f'visualize/{dataset}/{num}_gaussian_{g}.png')
            # scaling
            ss = [0.7, 0.9, 1.5]
            for s in ss:
                transforms.visualize(sizeScaleT.proc(ds[num][0], s), f'visualize/{dataset}/{num}_scale_{s}.png')
