
import os
import random
import math
import numpy as np
import torch
import torchvision
import PIL.Image
from torchvision.transforms import *
import torchvision.transforms.functional as TF
import cv2
import numba
from numba import jit
try:
    from semantic.C import transform_kern as kern
    print('Fast kernel loaded')
    use_kern = True
except Exception:
    print('Fast kernel not available, use PyTorch Kernel')
    use_kern = False

EPS = 1e-6

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

    def raw_proc(self, input: torch.Tensor, angle: float):
        if abs(angle) < EPS:
            return input
        if use_kern:
            np_input = np.ascontiguousarray(input.numpy(), dtype=np.float)
            np_output = kern.rotation(np_input, angle)
            output = torch.tensor(np_output)
            return output
        else:
            c, h, w = input.shape
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

            rows = torch.linspace(0.0, h - 1, steps=h)
            cols = torch.linspace(0.0, w - 1, steps=w)

            dist_mat = ((rows - cy) * (rows - cy)).unsqueeze(1) + ((cols - cx) * (cols - cx)).unsqueeze(0)
            dist_mat = torch.sqrt(dist_mat)

            rows_mat = rows.unsqueeze(1).repeat(1, w)
            cols_mat = cols.repeat(h, 1)
            alpha_mat = torch.atan2(rows_mat - cy, cols_mat - cx)
            beta_mat = alpha_mat + angle * math.pi / 180.0

            ny_mat, nx_mat = dist_mat * torch.sin(beta_mat) + cy, dist_mat * torch.cos(beta_mat) + cx
            nyl_mat, nxl_mat = torch.floor(ny_mat).type(torch.LongTensor), torch.floor(nx_mat).type(torch.LongTensor)
            nyp_mat, nxp_mat = nyl_mat + 1, nxl_mat + 1
            torch.clamp_(nyl_mat, min=0, max=h - 1)
            torch.clamp_(nxl_mat, min=0, max=w - 1)
            torch.clamp_(nyp_mat, min=0, max=h - 1)
            torch.clamp_(nxp_mat, min=0, max=w - 1)

            nyb_cell, nxb_cell = torch.flatten(nyl_mat), torch.flatten(nxl_mat)
            nyp_cell, nxp_cell = torch.flatten(nyp_mat), torch.flatten(nxp_mat)

            Pll = torch.gather(input.reshape(c, h * w), dim=1, index=(nyb_cell * w + nxb_cell).repeat(c, 1)).reshape(c, h, w)
            Plr = torch.gather(input.reshape(c, h * w), dim=1, index=(nyb_cell * w + nxp_cell).repeat(c, 1)).reshape(c, h, w)
            Prl = torch.gather(input.reshape(c, h * w), dim=1, index=(nyp_cell * w + nxb_cell).repeat(c, 1)).reshape(c, h, w)
            Prr = torch.gather(input.reshape(c, h * w), dim=1, index=(nyp_cell * w + nxp_cell).repeat(c, 1)).reshape(c, h, w)

            nyl_mat, nxl_mat = nyl_mat.type(torch.FloatTensor), nxl_mat.type(torch.FloatTensor)

            raw = (ny_mat - nyl_mat) * (nx_mat - nxl_mat) * Prr + \
                  (ny_mat - nyl_mat) * (1.0 - nx_mat + nxl_mat) * Prl + \
                  (1.0 - ny_mat + nyl_mat) * (nx_mat - nxl_mat) * Plr + \
                  (1.0 - ny_mat + nyl_mat) * (1.0 - nx_mat + nxl_mat) * Pll
            out = raw
            return out

    def old_raw_proc(self, input, angle):
        pil = TF.to_pil_image(input)
        rot = TF.rotate(pil, angle, PIL.Image.BILINEAR)
        out = TF.to_tensor(rot)
        return out

    def masking(self, input: torch.Tensor):
        return input * self.mask

    def proc(self, input: torch.Tensor, angle: float):
        return self.masking(self.raw_proc(input, angle) if abs(angle) > EPS else input)

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


class BlackTranslational(Translational):

    def __init__(self, canopy, sigma):
        super(BlackTranslational, self).__init__(canopy, sigma)

    def proc(self, input, dx, dy):
        nx, ny = round(dx), round(dy)
        out = torch.zeros_like(input)
        nx = nx % self.h if nx > 0 else nx % (-self.h)
        ny = ny % self.w if ny > 0 else ny % (-self.w)
        if nx > 0 and ny > 0:
            out[:, :-nx, :-ny] = input[:, nx:, ny:]
        elif nx > 0 and ny == 0:
            out[:, :-nx, :] = input[:, nx:, :]
        elif nx > 0 and ny < 0:
            out[:, :-nx, -ny:] = input[:, nx:, :ny]
        elif nx == 0 and ny > 0:
            out[:, :, :-ny] = input[:, :, ny:]
        elif nx == 0 and ny == 0:
            out = input
        elif nx == 0 and ny < 0:
            out[:, :, -ny:] = input[:, :, :ny]
        elif nx < 0 and ny > 0:
            out[:, -nx:, :-ny] = input[:, :nx, ny:]
        elif nx < 0 and ny == 0:
            out[:, -nx:, :] = input[:, :nx, :]
        elif nx < 0 and ny < 0:
            out[:, -nx:, -ny:] = input[:, :nx, :ny]
        return out



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
        if abs(s - 1) < EPS:
            return input
        if use_kern:
            np_input = np.ascontiguousarray(input.numpy(), dtype=np.float)
            np_output = kern.scaling(np_input, s)
            output = torch.tensor(np_output)
            return output
        else:
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

            # Pll_old = torch.gather(torch.index_select(input, dim=1, index=nyl_arr), dim=2,
            #                        index=nxl_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
            # Plr_old = torch.gather(torch.index_select(input, dim=1, index=nyl_arr), dim=2,
            #                        index=nxr_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
            # Prl_old = torch.gather(torch.index_select(input, dim=1, index=nyr_arr), dim=2,
            #                        index=nxl_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)
            # Prr_old = torch.gather(torch.index_select(input, dim=1, index=nyr_arr), dim=2,
            #                        index=nxr_arr.repeat(c, 1).unsqueeze(2)).reshape(c, h, w)

            Pll = torch.gather(input.reshape(c, h * w), dim=1, index=(nxl_arr + nyl_arr * w).repeat(c, 1)).reshape(c, h, w)
            Plr = torch.gather(input.reshape(c, h * w), dim=1, index=(nxr_arr + nyl_arr * w).repeat(c, 1)).reshape(c, h, w)
            Prl = torch.gather(input.reshape(c, h * w), dim=1, index=(nxl_arr + nyr_arr * w).repeat(c, 1)).reshape(c, h, w)
            Prr = torch.gather(input.reshape(c, h * w), dim=1, index=(nxr_arr + nyr_arr * w).repeat(c, 1)).reshape(c, h, w)

            # print(torch.sum(torch.abs(Pll - Pll_old)))
            # print(torch.sum(torch.abs(Plr - Plr_old)))
            # print(torch.sum(torch.abs(Prl - Prl_old)))
            # print(torch.sum(torch.abs(Prr - Prr_old)))

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


class Gaussian:
    # it adopts uniform distribution
    def __init__(self, sigma):
        self.sigma = sigma
        self.sigma2 = sigma ** 2.0

    def gen_param(self):
        r = random.uniform(0.0, self.sigma2)
        return r

    def proc(self, input, r2):
        out = cv2.GaussianBlur(input.numpy().transpose(1, 2, 0), (0, 0), math.sqrt(r2), borderType=cv2.BORDER_REFLECT101)
        if out.ndim == 2:
            out = np.expand_dims(out, 2)
        out = torch.from_numpy(out.transpose(2, 0, 1))
        return out

    def batch_proc(self, inputs):
        outs = torch.zeros_like(inputs)
        for i in range(len(inputs)):
            outs[i] = self.proc(inputs[i], self.gen_param())
        return outs

class ExpGaussian(Gaussian):
    # it adopts exponential distribution
    # where the sigma is actually lambda in exponential distribution Exp(1/lambda)
    def __init__(self, sigma):
        super(ExpGaussian, self).__init__(sigma)

    def gen_param(self):
        r = - self.sigma * math.log(random.uniform(0.0, 1.0))
        return r

class FoldGaussian(Gaussian):
    def __init__(self, sigma):
        super(FoldGaussian, self).__init__(sigma)

    def gen_param(self):
        r = abs(random.normalvariate(0.0, self.sigma))
        return r


def visualize(img, outfile):
    img = torch.tensor(img).clamp_(min=0.0, max=1.0)
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    torchvision.utils.save_image(img, outfile, range=(0.0, 1.0))


