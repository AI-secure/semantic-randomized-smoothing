
import semantic.transforms as transforms
from scipy.stats import norm


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

    def __init__(self, sigma, canopy):
        super(RotationNoiseTransformer, self).__init__()
        self.sigma = sigma
        self.noise_adder = transforms.Noise(self.sigma)
        self.rotation_adder = transforms.Rotation(canopy)
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

    def __init__(self, canopy):
        super(RotationTransformer, self).__init__()
        self.rotation_adder = transforms.Rotation(canopy)
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


def gen_transformer(args, canopy) -> AbstractTransformer:
    if args.transtype == 'rotation-noise':
        return RotationNoiseTransformer(args.noise_sd, canopy)
    elif args.transtype == 'rotation':
        return RotationTransformer(canopy)
    elif args.transtype == 'noise':
        return NoiseTransformer(args.noise_sd)
    elif args.transtype == 'strict-rotation-noise':
        rnt = RotationNoiseTransformer(args.noise_sd, canopy)
        rnt.set_round(1)
        return rnt
    else:
        raise NotImplementedError
