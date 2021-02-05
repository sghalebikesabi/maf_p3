import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt

import datasets
import util


class CUSTOM:
    """
    Custom data set class for weighted DP analysis.
    """

    alpha = 1.0e-6

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data, logit, dequantize, rng):

            x = self._dequantize(data[0], rng) if dequantize else data[0]  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x              # logit
            self.labels = data[1]                                          # numeric labels
            self.y = util.one_hot_encode(self.labels, 2)                  # 1-hot encoded labels
            self.N = self.x.shape[0]                                       # number of datapoints

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        @staticmethod
        def _logit_transform(x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            return util.logit(CUSTOM.alpha + (1 - 2*CUSTOM.alpha) * x)

    def __init__(
        self, 
        data_trn, 
        data_val, 
        labels_trn, 
        labels_val, 
        logit=True, 
        dequantize=False):

        # load dataset
        trn, val = (data_trn, labels_trn), (data_val, labels_val)

        rng = np.random.RandomState(42)
        self.trn = self.Data(trn, logit, dequantize, rng)
        self.val = self.Data(val, logit, dequantize, rng)
        # self.tst = self.Data(tst, logit, dequantize, rng)

        self.n_dims = self.trn.x.shape[1]
        self.n_labels = self.trn.y.shape[1]

    def show_histograms(self, split, vars):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        util.plot_hist_marginals(data_split.x[:, vars])
        plt.show()

    def show_images(self, split):
        """
        Displays the images in a given split.
        :param split: string
        """

        # get split
        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        # display images
        util.disp_imdata(data_split.x, self.image_size, [6, 10])

        plt.show()
