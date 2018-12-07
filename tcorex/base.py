from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from tcorex.experiments.misc import make_sure_path_exists
from scipy.stats import norm, rankdata
import numpy as np
import time
import torch
import os


def g(x, t=4):
    """A transformation that suppresses outliers for a standard normal."""
    xp = np.clip(x, -t, t)
    diff = np.tanh(x - xp)
    return xp + diff


def g_inv(x, t=4):
    """Inverse of g transform."""
    xp = np.clip(x, -t, t)
    diff = np.arctanh(np.clip(x - xp, -1 + 1e-10, 1 - 1e-10))
    return xp + diff


def mean_impute(x, v):
    """Missing values in the data, x, are indicated by v. Wherever this value appears in x, it is replaced by the
    mean value taken from the marginal distribution of that column."""
    if not np.isnan(v):
        x = np.where(x == v, np.nan, x)
    x_new = []
    n_obs = []
    for i, xi in enumerate(x.T):
        missing_locs = np.where(np.isnan(xi))[0]
        xi_nm = xi[np.isfinite(xi)]
        xi[missing_locs] = np.mean(xi_nm)
        x_new.append(xi)
        n_obs.append(len(xi_nm))
    return np.array(x_new).T, np.array(n_obs)


def to_numpy(x):
    if x.requires_grad:
        x = x.detach()
    if x.device.type != 'cpu':
        x = x.cpu()
    return x.numpy()


def save(model, path, verbose=False):
    save_dir = os.path.dirname(path)
    make_sure_path_exists(save_dir)
    if verbose:
        print('Saving into {}'.format(path))
    torch.save(model, path)


def load(path):
    return torch.load(path)


class TCorexBase(object):
    def __init__(self, nt, nv, n_hidden=10, max_iter=10000, tol=1e-5, anneal=True,
                 missing_values=None, gaussianize='standard', y_scale=1.0,
                 pretrained_weights=None, device='cpu', verbose=0):
        self.nt = nt  # Number of timesteps
        self.nv = nv  # Number of variables
        self.m = n_hidden  # Number of latent factors to learn
        self.max_iter = max_iter  # Number of iterations to try
        self.tol = tol  # Threshold for convergence
        self.anneal = anneal
        self.eps = 0  # If anneal is True, it's adjusted during optimization to avoid local minima
        self.missing_values = missing_values
        self.gaussianize = gaussianize  # Preprocess data: 'standard' scales to zero mean and unit variance
        self.y_scale = y_scale  # Can be arbitrary, but sets the scale of Y
        self.pretrained_weights = pretrained_weights
        self.device = torch.device(device)
        self.verbose = verbose
        if verbose:
            np.set_printoptions(precision=3, suppress=True, linewidth=160)
            print('Linear CorEx with {:d} latent factors'.format(n_hidden))
        self.add_params = []  # used in _train_loop()

    def forward(self, x_wno, anneal_eps, indices=None):
        raise NotImplementedError("forward function should be specified for all child classes")

    def _train_loop(self):
        """ train loop expects self have one variable x_input
        """
        if self.verbose:
            print("Starting the training loop ...")
        # set the annealing schedule
        anneal_schedule = [0.]
        if self.anneal:
            anneal_schedule = [0.6 ** k for k in range(1, 7)] + [0]

        # initialize the weights if pre-trained weights are specified
        if self.pretrained_weights is not None:
            self.load_weights(self.pretrained_weights)

        # set up the optimizer
        optimizer = torch.optim.Adam(self.ws + self.add_params)

        for eps in anneal_schedule:
            start_time = time.time()
            self.eps = eps  # for Greg's part of code
            for i_loop in range(self.max_iter):
                # TODO: write a stopping condition
                ret = self.forward(self.x_input, eps)
                obj = ret['total_obj']

                optimizer.zero_grad()
                obj.backward()
                optimizer.step()

                main_obj = ret['main_obj']
                reg_obj = ret['reg_obj']
                if self.verbose > 1:
                    print("iter: {} / {}, obj: {:.4f}, main: {:.4f}, reg: {:.4f}, eps: {:.4f}".format(
                        i_loop, self.max_iter, obj, main_obj, reg_obj, eps), end='\r')
            print("Annealing iteration finished, time = {}".format(time.time() - start_time))

        # clear cache to free some GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return self

    def fit(self, x):
        x = [np.array(xt, dtype=np.float32) for xt in x]
        x = self.preprocess(x, fit=True)  # fit a transform for each marginal
        self.x_input = x  # to have an access to input
        return self._train_loop()

    def get_weights(self):
        return [to_numpy(w) for w in self.ws]

    @property
    def mis(self):
        """ Returns I (Z_j : X_i) for each time period. """
        R = self.forward(self.x_input, 0)['R']
        R = [to_numpy(rho) for rho in R]
        return [-0.5 * np.log1p(rho ** 2) for rho in R]

    def clusters(self, type='MI'):
        """ Get clusters of variables for each time period.
        :param type: MI or W. In case of MI, the cluster is defined as argmax_j I(x_i : z_j).
                     In case of W, the cluster is defined as argmax_j |W_{j,i}|
        """
        if type == 'W':
            return [np.abs(w).argmax(axis=0) for w in self.get_weights()]
        return [mi.argmax(axis=0) for mi in self.mis]

    def transform(self, x):
        """ Transform an array of inputs, x, into an array of k latent factors, Y. """
        x = self.preprocess(x)
        ws = self.get_weights()
        ret = [a.dot(w.T) for (a, w) in zip(x, ws)]
        return ret

    def preprocess(self, X, fit=False):
        """Transform each marginal to be as close to a standard Gaussian as possible.
        'standard' (default) just subtracts the mean and scales by the std.
        'empirical' does an empirical gaussianization (but this cannot be inverted).
        'outliers' tries to squeeze in the outliers
        Any other choice will skip the transformation."""
        warnings = []
        ret = [None] * len(X)
        if fit:
            self.theta = []
        for t in range(len(X)):
            x = X[t]
            if self.missing_values is not None:
                x, n_obs = mean_impute(x, self.missing_values)  # Creates a copy
            else:
                n_obs = len(x)
            if self.gaussianize == 'none':
                pass
            elif self.gaussianize == 'standard':
                if fit:
                    mean = np.mean(x, axis=0)
                    # std = np.std(x, axis=0, ddof=0).clip(1e-10)
                    std = np.sqrt(np.sum((x - mean) ** 2, axis=0) / n_obs).clip(1e-10)
                    self.theta.append((mean, std))
                x = ((x - self.theta[t][0]) / self.theta[t][1])
                if np.max(np.abs(x)) > 6 and self.verbose:
                    warnings.append("Warning: outliers more than 6 stds away from mean. "
                                    "Consider using gaussianize='outliers'")
            elif self.gaussianize == 'outliers':
                if fit:
                    mean = np.mean(x, axis=0)
                    std = np.std(x, axis=0, ddof=0).clip(1e-10)
                    self.theta.append((mean, std))
                x = g((x - self.theta[t][0]) / self.theta[t][1])  # g truncates long tails
            elif self.gaussianize == 'empirical':
                warnings.append("Warning: correct inversion/transform of empirical gauss transform not implemented.")
                x = np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T
            ret[t] = x
        for w in set(warnings):
            print(w)
        return ret

    def get_covariance(self, indices=None, normed=False):
        if indices is None:
            indices = range(self.nt)
        cov = self.forward(self.x_input, anneal_eps=0, indices=indices)['sigma']
        for t in range(self.nt):
            if cov[t] is None:
                continue
            cov[t] = to_numpy(cov[t])
            if not normed:
                cov[t] = self.theta[t][1][:, np.newaxis] * self.theta[t][1] * cov[t]
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return cov

    def load_weights(self, weights):
        self.ws = [torch.tensor(w, dtype=torch.float, device=self.device, requires_grad=True)
                   for w in weights]