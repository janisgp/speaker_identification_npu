"""
Feature extraction using Yaafe
------------------------------
"""

import yaafelib
import numpy as np
import scipy.io.wavfile as wav


class YaafeFeatureExtractor(object):
    """Base feature extraction. Should not be used directly.

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    stack : int, optional
        Stack `stack` consecutive features. Defaults to 1.
    """

    def __init__(self, sample_rate=16000, duration=0.025, step=0.010, stack=1):

        super(YaafeFeatureExtractor, self).__init__()

        self.sample_rate = sample_rate
        self.duration = duration
        self.step = step
        self.stack = stack

        start = -0.5 * self.duration

        self.engine_ = yaafelib.Engine()

    def dimension(self):
        raise NotImplementedError('')

    def sliding_window(self):
        return self.sliding_window_

    def __call__(self, path):
        """Extract features

        Parameters
        ----------
        path : path to .wav file

        Returns
        -------
        data : numpy array

        """

        # --- load audio file

        sample_rate, y = wav.read(path)

        # --- update data_flow every time sample rate changes
        if not hasattr(self, 'sample_rate_') or self.sample_rate_ != sample_rate:
            self.sample_rate_ = sample_rate
            feature_plan = yaafelib.FeaturePlan(sample_rate=self.sample_rate_)
            for name, recipe in self.definition():
                assert feature_plan.addFeature(
                    "{name}: {recipe}".format(name=name, recipe=recipe))
            data_flow = feature_plan.getDataFlow()
            self.engine_.load(data_flow)

        # Yaafe needs this: float64, column-contiguous, 2-dimensional
        y = np.array(y, dtype=np.float64, order='C').reshape((1, -1))

        # --- extract features
        features = self.engine_.processAudio(y)
        data = np.hstack([features[name] for name, _ in self.definition()])

        # --- stack features
        n_samples, n_features = data.shape
        zero_padding = self.stack // 2
        if self.stack % 2 == 0:
            expanded_data = np.concatenate(
                (np.zeros((zero_padding, n_features)) + data[0],
                 data,
                 np.zeros((zero_padding - 1, n_features)) + data[-1]))
        else:
            expanded_data = np.concatenate((
                np.zeros((zero_padding, n_features)) + data[0],
                data,
                np.zeros((zero_padding, n_features)) + data[-1]))

        data = np.lib.stride_tricks.as_strided(
            expanded_data,
            shape=(n_samples, n_features * self.stack),
            strides=data.strides)

        self.engine_.reset()

        return data


class YaafeMFCC(YaafeFeatureExtractor):
    """MFCC feature extraction

    ::

            | e    |  energy
            | c1   |
            | c2   |  coefficients
            | c3   |
            | ...  |
            | Δe   |  energy first derivative
            | Δc1  |
        x = | Δc2  |  coefficients first derivatives
            | Δc3  |
            | ...  |
            | ΔΔe  |  energy second derivative
            | ΔΔc1 |
            | ΔΔc2 |  coefficients second derivatives
            | ΔΔc3 |
            | ...  |

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000.
    duration : float, optional
        Defaults to 0.025.
    step : float, optional
        Defaults to 0.010.
    e : bool, optional
        Energy. Defaults to True.
    coefs : int, optional
        Number of coefficients. Defaults to 11.
    De : bool, optional
        Keep energy first derivative. Defaults to False.
    D : bool, optional
        Add first order derivatives. Defaults to False.
    DDe : bool, optional
        Keep energy second derivative. Defaults to False.
    DD : bool, optional
        Add second order derivatives. Defaults to False.

    Notes
    -----
    Default Yaafe values:
        * fftWindow = Hanning
        * melMaxFreq = 6854.0
        * melMinFreq = 130.0
        * melNbFilters = 40

    """

    def __init__(
            self, sample_rate=16000, duration=0.025, step=0.010, stack=1,
            e=True, coefs=11, De=False, DDe=False, D=False, DD=False):

        self.e = e
        self.coefs = coefs
        self.De = De
        self.DDe = DDe
        self.D = D
        self.DD = DD

        super(YaafeMFCC, self).__init__(sample_rate=sample_rate,
                                        duration=duration, step=step,
                                        stack=stack)

    def dimension(self):

        n_features = 0
        n_features += self.e
        n_features += self.De
        n_features += self.DDe
        n_features += self.coefs
        n_features += self.coefs * self.D
        n_features += self.coefs * self.DD

        return n_features * self.stack

    def definition(self):

        blockSize = int(self.sample_rate_ * self.duration)
        stepSize = int(self.sample_rate_ * self.step)

        d = []

        # --- coefficients
        # 0 if energy is kept
        # 1 if energy is removed
        d.append((
            "mfcc",
            "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
            "blockSize=%d stepSize=%d" % (
                0 if self.e else 1,
                self.coefs + self.e * 1,
                blockSize, stepSize
            )))

        # --- 1st order derivatives
        if self.De or self.D:
            d.append((
                "mfcc_d",
                "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                "blockSize=%d stepSize=%d > Derivate DOrder=1" % (
                    0 if self.De else 1,
                    self.D * self.coefs + self.De * 1,
                    blockSize, stepSize
                )))

        # --- 2nd order derivatives
        if self.DDe or self.DD:
            d.append((
                "mfcc_dd",
                "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                "blockSize=%d stepSize=%d > Derivate DOrder=2" % (
                    0 if self.DDe else 1,
                    self.DD * self.coefs + self.DDe * 1,
                    blockSize, stepSize
                )))

        return d