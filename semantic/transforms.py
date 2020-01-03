
import os
import random
import torch
import torchvision
import PIL.Image
from torchvision.transforms import *
import torchvision.transforms.functional as TF

class Noise:
    def __init__(self, sigma):
        self.sigma = sigma

    def proc(self, input):
        noise = torch.randn_like(input) * self.sigma
        return input + noise

    def batch_proc(self, inputs):
        noise = torch.randn_like(inputs) * self.sigma
        return inputs + noise


class Rotation:
    def __init__(self, canopy):
        self.h = canopy.shape[-2]
        self.w = canopy.shape[-1]
        assert self.h == self.w
        self.mask = torch.ones((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                if (i - (self.h-1)/2.0) ** 2 + (j - (self.w-1)/2.0) ** 2 > ((self.h-1)/2.0) ** 2:
                    self.mask[i][j] = 0
        self.mask.unsqueeze_(0)

    def gen_param(self):
        return random.uniform(0, 360)

    def raw_proc(self, input, angle):
        pil = TF.to_pil_image(input)
        rot = TF.rotate(pil, angle, PIL.Image.BILINEAR)
        out = TF.to_tensor(rot)
        return out

    def masking(self, input):
        return input * self.mask

    def proc(self, input, angle):
        return self.masking(self.raw_proc(input, angle))

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs

    def batch_masking(self, inputs):
        return inputs * self.mask.unsqueeze(0)


class Translational:

    def __init__(self, sigma):
        self.sigma = sigma

    def gen_param(self):
        tx, ty = torch.randn(2)
        tx, ty = tx.item(), ty.item()
        return tx * self.sigma, ty * self.sigma

    def proc(self, input, dx, dy):
        nx, ny = round(dx), round(dy)
        # TODO
        pass

    def batch_proc(self, inputs):
        pass


class BrightnessShift:

    def __init__(self, sigma):
        pass

    def gen_param(self):
        pass

    def proc(self, ds):
        pass

    def batch_proc(self, inputs):
        pass


class BrightnessScale:

    def __init__(self, sigma):
        pass

    def gen_param(self):
        pass

    def proc(self, dk):
        # scale by exp(dk)
        pass

    def batch_proc(self, inputs):
        pass


def visualize(img, outfile):
    img = torch.tensor(img)
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    torchvision.utils.save_image(img, outfile, range=(0.0, 1.0))


