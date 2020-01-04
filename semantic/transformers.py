
import semantic.transforms as transforms
from scipy.stats import norm
import math


class AbstractTransformer:

    def process(self, inputs):
        raise NotImplementedError

    def calc_radius(self, pABar: float) -> float:
        return 0.0


class NoiseTransformer(AbstractTransformer):

    def __init__(self, sigma):
        super(NoiseTransformer, self).__init__()
        self.sigma = sigma
        self.noise_adder = transforms.Noise(self.sigma)

    def process(self, inputs):
        outs = self.noise_adder.batch_proc(inputs)
        return outs

    def calc_radius(self, pABar: float):
        radius = self.sigma * norm.ppf(pABar)
        return radius


class RotationNoiseTransformer(AbstractTransformer):

    def __init__(self, sigma, canopy, rotation_angle=180.0):
        super(RotationNoiseTransformer, self).__init__()
        self.sigma = sigma
        self.noise_adder = transforms.Noise(self.sigma)
        self.rotation_adder = transforms.Rotation(canopy, rotation_angle)
        self.round = 2
        self.masking = True

    def set_round(self, r=1):
        self.round = r

    def enable_masking(self, masking):
        self.masking = masking

    def process(self, inputs):
        # two-round rotation for training & certifying
        # for predicting, only one-round rotation
        # then add Gaussian noise
        outs = inputs
        for r in range(self.round):
            outs = self.rotation_adder.batch_proc(outs)
        outs = self.noise_adder.batch_proc(outs)
        if self.masking:
            outs = self.rotation_adder.batch_masking(outs)
        return outs

    def calc_radius(self, pABar: float):
        radius = self.sigma * norm.ppf(pABar)
        return radius


class RotationTransformer(AbstractTransformer):

    def __init__(self, canopy, rotation_angle=180.0):
        super(RotationTransformer, self).__init__()
        self.rotation_adder = transforms.Rotation(canopy, rotation_angle)
        self.round = 2

    def set_round(self, r=1):
        self.round = r

    def process(self, inputs):
        # only two-round rotation for training & certifying
        # for predicting, only one-round rotation
        outs = inputs
        for r in range(self.round):
            outs = self.rotation_adder.batch_proc(outs)
        return outs

    def calc_radius(self, pABar: float):
        # return infinity
        return 1e+99


class TranslationTransformer(AbstractTransformer):

    def __init__(self, sigma, canopy):
        super(TranslationTransformer, self).__init__()
        self.translation_adder = transforms.Translational(canopy, sigma)
        self.sigma = sigma

    def process(self, inputs):
        outs = self.translation_adder.batch_proc(inputs)
        return outs

    def calc_radius(self, pABar: float):
        radius = self.sigma * norm.ppf(pABar)
        return radius


class BrightnessTransformer(AbstractTransformer):

    def __init__(self, sigma_k, sigma_b):
        super(BrightnessTransformer, self).__init__()
        self.sigma_k = sigma_k
        self.sigma_b = sigma_b
        self.scaler = transforms.BrightnessScale(sigma_k)
        self.brighter = transforms.BrightnessShift(sigma_b)
        self.k_l = self.k_r = 0

    def process(self, inputs):
        outs = self.scaler.batch_proc(self.brighter.batch_proc(inputs))
        return outs

    def set_brightness_scale(self, l, r):
        self.k_l = math.log(l)
        self.k_r = math.log(r)

    def calc_radius(self, pABar: float):
        return min(self.calc_b_bound(self.k_l, pABar), self.calc_b_bound(self.k_r, pABar))

    def calc_b_bound(self, k: float, pABar: float):
        if k >= 0.0:
            pBBar = 2.0 - 2.0 * norm.cdf(math.exp(k / 2.0) * norm.ppf(1.0 - pABar / 2.0))
        else:
            pBBar = 2.0 * norm.cdf(math.exp(k / 2.0) * norm.ppf(0.5 + pABar / 2.0)) - 1.0

        if pBBar > 0.5:
            margin = norm.ppf(pBBar) ** 2 - (k / self.sigma_k) ** 2
            if margin > 0.0:
                return self.sigma_b * math.sqrt(margin)
        return 0.0


class ResizeTransformer(AbstractTransformer):

    def __init__(self, sl, sr):
        super(ResizeTransformer, self).__init__()
        self.sl, self.sr = sl, sr
        self.resizer = transforms.Resize(self.sl, self.sr)

    def process(self, inputs):
        outs = self.resizer.batch_proc(inputs)
        return outs

    def calc_radius(self, pABar: float):
        # return infinity
        return 1e+99


def gen_transformer(args, canopy) -> AbstractTransformer:
    if args.transtype == 'rotation-noise':
        print(f'rotation-noise with noise {args.noise_sd}')
        return RotationNoiseTransformer(args.noise_sd, canopy)
    elif args.transtype == 'rotation':
        print(f'rotation')
        return RotationTransformer(canopy)
    elif args.transtype == 'noise':
        print(f'noise {args.noise_sd}')
        return NoiseTransformer(args.noise_sd)
    elif args.transtype == 'strict-rotation-noise':
        print(f'strict rotation angle in +-{args.rotation_angle} and noise in {args.noise_sd}')
        rnt = RotationNoiseTransformer(args.noise_sd, canopy, args.rotation_angle)
        rnt.set_round(1)
        return rnt
    elif args.transtype == 'translation':
        print(f'translation with noise {args.noise_sd}')
        return TranslationTransformer(args.noise_sd, canopy)
    elif args.transtype == 'brightness':
        print(f'brightness with k noise {args.noise_k} and b noise {args.noise_b}')
        return BrightnessTransformer(args.noise_k, args.noise_b)
    else:
        raise NotImplementedError
