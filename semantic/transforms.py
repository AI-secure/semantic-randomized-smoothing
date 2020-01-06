
import os
import random
import math
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
    def __init__(self, canopy, rotation_angle):
        self.h = canopy.shape[-2]
        self.w = canopy.shape[-1]
        assert self.h == self.w
        self.mask = torch.ones((self.h, self.w))
        for i in range(self.h):
            for j in range(self.w):
                if (i - (self.h-1)/2.0) ** 2 + (j - (self.w-1)/2.0) ** 2 > ((self.h-1)/2.0) ** 2:
                    self.mask[i][j] = 0
        self.mask.unsqueeze_(0)
        self.rotation_angle = rotation_angle

    def gen_param(self):
        return random.uniform(-self.rotation_angle, self.rotation_angle)

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

    def __init__(self, canopy, sigma):
        self.sigma = sigma
        self.c, self.h, self.w = canopy.shape

    def gen_param(self):
        tx, ty = torch.randn(2)
        tx, ty = tx.item(), ty.item()
        return tx * self.sigma, ty * self.sigma

    def proc(self, input, dx, dy):
        nx, ny = round(dx), round(dy)
        nx, ny = nx % self.h, ny % self.w
        out = torch.zeros_like(input)
        if nx > 0 and ny > 0:
            out[:, -nx:, -ny:] = input[:, :nx, :ny]
            out[:, -nx:, :-ny] = input[:, :nx, ny:]
            out[:, :-nx, -ny:] = input[:, nx:, :ny]
            out[:, :-nx, :-ny] = input[:, nx:, ny:]
        elif ny > 0:
            out[:, :, -ny:] = input[:, :, :ny]
            out[:, :, :-ny] = input[:, :, ny:]
        elif nx > 0:
            out[:, -nx:, :] = input[:, :nx, :]
            out[:, :-nx, :] = input[:, nx:, :]
        else:
            out = input
        return out

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], *self.gen_param())
        return outs


class BrightnessShift:

    def __init__(self, sigma):
        self.sigma = sigma

    def gen_param(self):
        d = torch.randn(1).item() * self.sigma
        return d

    def proc(self, input, d):
        # print(d)
        return input + d

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs


class BrightnessScale:

    def __init__(self, sigma):
        self.sigma = sigma

    def gen_param(self):
        d = torch.randn(1).item() * self.sigma
        return d

    def proc(self, input, dk):
        # scale by exp(dk)
        # print(dk)
        return input * math.exp(dk)

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs


class Resize:

    def __init__(self, canopy, sl, sr):
        self.sl, self.sr = sl, sr
        self.c, self.h, self.w = canopy.shape
        self.rows = torch.linspace(0.0, self.h - 1, steps=self.h)
        self.cols = torch.linspace(0.0, self.w - 1, steps=self.w)


    def gen_param(self):
        return random.uniform(self.sl, self.sr)

    def proc(self, input, s):
        c, h, w = self.c, self.h, self.w
        cy, cx = float(h - 1) / 2.0, float(w - 1) / 2.0
        nys = (self.rows - cy) / s + cy
        nxs = (self.cols - cx) / s + cx

        nysl, nxsl = torch.floor(nys), torch.floor(nxs)
        nysr, nxsr = nysl + 1, nxsl + 1

        nysl = nysl.clamp(min=0, max=h-1).type(torch.LongTensor)
        nxsl = nxsl.clamp(min=0, max=w-1).type(torch.LongTensor)
        nysr = nysr.clamp(min=0, max=h-1).type(torch.LongTensor)
        nxsr = nxsr.clamp(min=0, max=w-1).type(torch.LongTensor)

        nyl_mat, nyr_mat, ny_mat = nysl.unsqueeze(1).repeat(1, w), nysr.unsqueeze(1).repeat(1, w), nys.unsqueeze(1).repeat(1, w)
        nxl_mat, nxr_mat, nx_mat = nxsl.repeat(h, 1), nxsr.repeat(h, 1), nxs.repeat(h, 1)

        nyl_arr, nyr_arr, nxl_arr, nxr_arr = nyl_mat.flatten(), nyr_mat.flatten(), nxl_mat.flatten(), nxr_mat.flatten()

        imgymin = max(math.ceil((1 - s) * cy), 0)
        imgymax = min(math.floor((1 - s) * cy + s * (h - 1)), h - 1)
        imgxmin = max(math.ceil((1 - s) * cx), 0)
        imgxmax = min(math.floor((1 - s) * cx + s * (h - 1)), w - 1)

        Pll = torch.gather(torch.index_select(input, dim=1, index=nyl_arr), dim=2,
                           index=nxl_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
        Plr = torch.gather(torch.index_select(input, dim=1, index=nyl_arr), dim=2,
                           index=nxr_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
        Prl = torch.gather(torch.index_select(input, dim=1, index=nyr_arr), dim=2,
                           index=nxl_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
        Prr = torch.gather(torch.index_select(input, dim=1, index=nyr_arr), dim=2,
                           index=nxr_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)

        nxl_mat, nyl_mat = nxl_mat.type(torch.FloatTensor), nyl_mat.type(torch.FloatTensor)

        out = torch.zeros_like(input)
        out[:, imgymin: imgymax + 1, imgxmin: imgxmax + 1] = (
            (ny_mat - nyl_mat) * (nx_mat - nxl_mat) * Prr +
            (1.0 - ny_mat + nyl_mat) * (nx_mat - nxl_mat) * Plr +
            (ny_mat - nyl_mat) * (1.0 - nx_mat + nxl_mat) * Prl +
            (1.0 - ny_mat + nyl_mat) * (1.0 - nx_mat + nxl_mat) * Pll)[:, imgymin: imgymax + 1, imgxmin: imgxmax + 1]

        return out

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs


def visualize(img, outfile):
    img = torch.tensor(img).clamp_(min=0.0, max=1.0)
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    torchvision.utils.save_image(img, outfile, range=(0.0, 1.0))


