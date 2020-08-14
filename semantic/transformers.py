
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
        # return infinity because it's invalid
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


class BlackpadTranslationTransformer(TranslationTransformer):

    def __init__(self, sigma, canopy):
        super(TranslationTransformer, self).__init__()
        self.translation_adder = transforms.BlackTranslational(canopy, sigma)
        self.sigma = sigma

    def calc_radius(self, pABar: float):
        # return infinity because it's invalid
        return 1e+99


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
            # Wrong!
            # pBBar = 2.0 - 2.0 * norm.cdf(math.exp(k / 2.0) * norm.ppf(1.0 - pABar / 2.0))
            pBBar = 2.0 - 2.0 * norm.cdf(math.exp(k) * norm.ppf(1.0 - pABar / 2.0))
        else:
            # Wrong!
            # pBBar = 2.0 * norm.cdf(math.exp(k / 2.0) * norm.ppf(0.5 + pABar / 2.0)) - 1.0
            pBBar = 2.0 * norm.cdf(math.exp(k) * norm.ppf(0.5 + pABar / 2.0)) - 1.0

        if pBBar > 0.5:
            if self.sigma_k == 0.0 and k == 0.0:
                margin = norm.ppf(pBBar) ** 2
            else:
                margin = norm.ppf(pBBar) ** 2 - (k / self.sigma_k) ** 2
            if margin > 0.0:
                return self.sigma_b * math.sqrt(margin)
        return 0.0

class ContrastTransformer(BrightnessTransformer):

    def __init__(self, sigma_k, sigma_b):
        super(ContrastTransformer, self).__init__(sigma_k, sigma_b)

    def process(self, inputs):
        outs = self.scaler.batch_proc(self.brighter.batch_proc(inputs))
        return outs

    def set_contrast_scale(self, l, r):
        self.k_l = math.log(l)
        self.k_r = math.log(r)
        assert self.k_l <= 0.0 <= self.k_r

    def calc_radius(self, pABar: float, EPS=1e-5):
        if pABar <= 0.5:
            return 0.0
        # binary left side
        l, r = self.k_l, 0.0
        while r - l > EPS:
            mid = (l + r) / 2.0
            if self.calc_b_bound(mid, pABar) > EPS:
                r = mid
            else:
                l = mid
        k_lbound = math.exp(r)
        # binary right side
        l, r = 0.0, self.k_r
        while r - l > EPS:
            mid = (l + r) / 2.0
            if self.calc_b_bound(mid, pABar) > EPS:
                l = mid
            else:
                r = mid
        k_rbound = math.exp(l)

        print('l', k_lbound, 'r', k_rbound)
        return min(1.0 - k_lbound, k_rbound - 1.0)


class ResizeTransformer(AbstractTransformer):

    def __init__(self, canopy, sl, sr):
        super(ResizeTransformer, self).__init__()
        self.sl, self.sr = sl, sr
        self.resizer = transforms.Resize(canopy, self.sl, self.sr)

    def process(self, inputs):
        outs = self.resizer.batch_proc(inputs)
        return outs

    def calc_radius(self, pABar: float):
        # return infinity
        return 1e+99


class ResizeNoiseTransformer(AbstractTransformer):

    def __init__(self, canopy, sl, sr, sigma):
        super(ResizeNoiseTransformer, self).__init__()
        self.resize_adder = transforms.Resize(canopy, sl, sr)
        self.noise_adder = transforms.Noise(sigma)

    def process(self, inputs):
        outs = self.noise_adder.batch_proc(self.resize_adder.batch_proc(inputs))
        return outs

    def calc_radius(self, pABar: float):
        # return infinity
        return 1e+99


class GaussianTransformer(AbstractTransformer):

    def __init__(self, sigma):
        super(GaussianTransformer, self).__init__()
        self.gaussian_adder = transforms.Gaussian(sigma)

    def process(self, inputs):
        outs = self.gaussian_adder.batch_proc(self.gaussian_adder.batch_proc(inputs))
        return outs

    def calc_radius(self, pABar: float):
        if pABar >= 0.5:
            return self.gaussian_adder.sigma * math.sqrt(pABar - 0.5)
        else:
            return 0.0


class ExpGaussianTransformer(AbstractTransformer):

    def __init__(self, sigma):
        super(ExpGaussianTransformer, self).__init__()
        self.gaussian_adder = transforms.ExpGaussian(sigma)

    def process(self, inputs):
        outs = self.gaussian_adder.batch_proc(self.gaussian_adder.batch_proc(inputs))
        return outs

    def calc_radius(self, pABar: float):
        if pABar >= 0.5:
            return math.sqrt(-self.gaussian_adder.sigma * math.log(2.0 - 2.0 * pABar))
        else:
            return 0.0


class RotationBrightnessNoiseTransformer(AbstractTransformer):

    def __init__(self, sigma, b, canopy, rotation_angle=180.0):
        super(RotationBrightnessNoiseTransformer, self).__init__()
        self.sigma = sigma
        self.b = b
        self.noise_adder = transforms.Noise(self.sigma)
        self.brightness_adder = transforms.BrightnessShift(self.b)
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
        outs = self.brightness_adder.batch_proc(outs)
        if self.masking:
            outs = self.rotation_adder.batch_masking(outs)
        return outs

    def calc_radius(self, pABar: float, b=0.):
        if norm.ppf(pABar) ** 2. - (b ** 2.) / (self.b ** 2.) >= 0.:
            radius = self.sigma * math.sqrt(norm.ppf(pABar) ** 2. - (b ** 2.) / (self.b ** 2.))
        else:
            radius = 0.
        return radius



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
    elif args.transtype == 'contrast':
        print(f'contrast with k noise {args.noise_k} and b noise {args.noise_b}')
        return ContrastTransformer(args.noise_k, args.noise_b)
    elif args.transtype == 'resize':
        print(f'resize from ratio {args.sl} to {args.sr} with noise {args.noise_sd}')
        return ResizeNoiseTransformer(canopy, args.sl, args.sr, args.noise_sd)
    elif args.transtype == 'expgaussian':
        print(f'gaussian with exponential noise from 0 to {args.noise_sd}')
        return ExpGaussianTransformer(args.noise_sd)
    elif args.transtype == 'gaussian':
        print(f'gaussian with uniform noise from 0 to {args.noise_sd}')
        return GaussianTransformer(args.noise_sd)
    elif args.transtype == 'btranslation':
        print(f'black-padding translation with noise {args.noise_sd}')
        return BlackpadTranslationTransformer(args.noise_sd, canopy)
    elif args.transtype == 'rotation-brightness':
        print(f'strict rotation angle in +-{args.rotation_angle} and noise in {args.noise_sd} with b noise {args.noise_b}')
        rnt = RotationBrightnessNoiseTransformer(args.noise_sd, args.noise_b, canopy, args.rotation_angle)
        rnt.set_round(1)
        return rnt
    else:
        raise NotImplementedError
