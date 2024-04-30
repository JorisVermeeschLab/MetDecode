from typing import Tuple, Any

import tqdm
import numpy as np
import scipy.optimize
import scipy.special
from scipy.optimize import nnls
from scipy.special import logit
import torch
import torch.nn.functional

from metdecode.optimizer import Optimizer


class MetDecode:

    def __init__(
            self,
            R_methylated: np.ndarray,
            R_depths: np.ndarray,
            X_methylated: np.ndarray,
            X_depths: np.ndarray,
            n_unknown_tissues: int = 0,
            coverage: bool = True,
            unsupervised: bool = True
    ):

        self.X_methylated: np.ndarray = X_methylated
        self.X_depths: np.ndarray = X_depths
        self.R_methylated: np.ndarray = R_methylated
        self.R_depths: np.ndarray = R_depths
        self.n_known_tissues: int = self.R_methylated.shape[0]
        self.n_unknown_tissues: int = n_unknown_tissues
        self.n_tissues: int = self.n_known_tissues + self.n_unknown_tissues
        self.coverage: bool = coverage
        self.unsupervised: bool = unsupervised

        # Add pseudo-counts
        self.R_methylated, self.R_depths = MetDecode.add_pseudo_counts(self.R_methylated, self.R_depths)
        self.X_methylated, self.X_depths = MetDecode.add_pseudo_counts(self.X_methylated, self.X_depths)

        R = np.asarray([self.R_methylated, self.R_depths - self.R_methylated])
        self.R: np.ndarray = np.transpose(R, (1, 2, 0)).reshape(self.R_depths.shape[0], 2 * self.R_depths.shape[1])
        X = np.asarray([self.X_methylated, self.X_depths - self.X_methylated])
        self.X: np.ndarray = np.transpose(X, (1, 2, 0)).reshape(self.X_depths.shape[0], 2 * self.X_depths.shape[1])

        self.info = {
            'loss': []
        }

    @staticmethod
    def add_pseudo_counts(methylated, depths, pc=0.8):
        methylated = np.asarray(methylated, dtype=float)
        depths = np.asarray(depths, dtype=float)
        avg_meth = np.sum(methylated) / np.sum(depths)

        # Compute per-row methylation
        row_methylated = np.sum(methylated, axis=1)
        row_depths = np.sum(depths, axis=1)
        mask = np.logical_or(row_methylated == 0, row_methylated == row_depths)
        row_depths[mask] += pc
        row_methylated[mask] += pc * avg_meth
        row_meth = row_methylated / row_depths

        # Compute per-column methylation
        col_methylated = np.sum(methylated, axis=0)
        col_depths = np.sum(depths, axis=0)
        mask = np.logical_or(col_methylated == 0, col_methylated == col_depths)
        col_depths[mask] += pc
        col_methylated[mask] += pc * avg_meth
        col_meth = col_methylated / col_depths

        # Impute missing values
        meth_prior = np.outer(np.ones(len(row_meth)), col_meth)  # TODO: use survival function of Beta distribution instead
        meth_prior = np.clip(meth_prior, 0.01, 0.99)

        mask = np.logical_or(methylated == 0, methylated == depths)
        methylated[mask] += pc * meth_prior[mask]
        depths[mask] += pc

        assert np.all(methylated > 0)
        assert not np.any(np.isnan(methylated))
        assert not np.any(np.isnan(depths))
        assert not np.any(np.isinf(methylated))
        assert not np.any(np.isinf(depths))
        assert np.all(methylated < depths)

        return methylated, depths

    @staticmethod
    def nnls(R_atlas: np.ndarray, R_cfdna: np.ndarray, W_cfdna: np.ndarray) -> np.ndarray:
        n_samples = len(R_cfdna)
        alpha_hat = []
        for i in range(n_samples):
            A = (np.sqrt(W_cfdna[np.newaxis, i, :]) * R_atlas).T
            b = np.sqrt(W_cfdna[i, :]) * R_cfdna[i, :]
            # x, residuals = nnls(A, b, maxiter=3000)
            x, residuals = nnls(A, b)
            alpha_hat.append(x)
        alpha_hat = np.asarray(alpha_hat)
        alpha_hat /= alpha_hat.sum(axis=1)[:, np.newaxis]
        return alpha_hat

    def _init_alpha_and_gamma(self) -> Tuple[np.ndarray, np.ndarray]:
        R_atlas = self.R_methylated / self.R_depths
        R_cfdna = self.X_methylated / self.X_depths
        W_cfdna = self.X_depths / np.sum(self.X_depths)
        alpha_hat = MetDecode.nnls(R_atlas, R_cfdna, W_cfdna)
        alpha_hat += 1e-6
        alpha_hat /= np.sum(alpha_hat, axis=1)[:, np.newaxis]

        #lb = np.min(R_atlas, axis=0)
        #ub = np.max(R_atlas, axis=0)
        lb = np.quantile(R_atlas, 0.4, axis=0)
        ub = np.quantile(R_atlas, 0.6, axis=0)

        n_unknowns = self.n_unknown_tissues
        while n_unknowns > 0:
            residuals = np.dot(alpha_hat, R_atlas) - R_cfdna
            residuals = np.median(residuals, axis=0)
            r = (residuals <= 0).astype(float)
            r = lb + r * (ub - lb)
            R_atlas = np.concatenate((R_atlas, r[np.newaxis, :]), axis=0)

            alpha_hat = MetDecode.nnls(R_atlas, R_cfdna, W_cfdna)
            alpha_hat += 1e-6
            alpha_hat /= np.sum(alpha_hat, axis=1)[:, np.newaxis]

            n_unknowns -= 1

        alpha_logit = np.log(alpha_hat)
        gamma_logit = logit(R_atlas)

        return alpha_logit, gamma_logit

    def deconvolute(self, max_n_iter: int = 2000, patience: int = 1000) -> np.ndarray:

        self.info['loss'] = []

        _alpha_logit, _gamma_logit = self._init_alpha_and_gamma()
        alpha_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(_alpha_logit)))
        gamma_logit = torch.nn.Parameter(torch.FloatTensor(np.copy(_gamma_logit)))
        gamma = torch.sigmoid(gamma_logit)

        R_atlas = scipy.special.expit(_gamma_logit)
        R_cfdna = self.X_methylated / self.X_depths

        R_cfdna = torch.FloatTensor(R_cfdna)

        optimizer = Optimizer(verbose=False)
        optimizer.add([alpha_logit], 1e-2)
        if self.unsupervised:
            optimizer.add([gamma_logit], 1e-5)

        n_steps_without_improvement = 0
        best_loss = np.inf
        extra_info = {'loss': []}
        for iteration in tqdm.tqdm(range(max_n_iter)):

            optimizer.zero_grad()

            alpha = torch.exp(alpha_logit)
            #alpha = torch.softmax(alpha_logit, dim=1)
            gamma = torch.sigmoid(gamma_logit)
            X_reconstructed = torch.mm(alpha, gamma)
            if self.coverage:
                weights = torch.sqrt(torch.FloatTensor(self.X_depths))
                weights = weights / torch.mean(weights)
                loss = torch.mean(weights * torch.abs(X_reconstructed - R_cfdna))
            else:
                loss = torch.mean(torch.abs(X_reconstructed - R_cfdna))
            loss.backward()

            self.info['loss'].append(loss.item())
            extra_info['loss'].append(loss.item())

            # Update parameters
            optimizer.step(loss.item())

            # scheduler.step()
            if loss.item() >= best_loss:
                n_steps_without_improvement += 1
                if n_steps_without_improvement >= patience:
                    break
            else:
                best_loss = loss.item()
                n_steps_without_improvement = 0

        alpha = scipy.special.softmax(alpha_logit.data.numpy(), axis=1)
        gamma = gamma.cpu().data.numpy()

        C = scipy.stats.spearmanr(gamma, R_atlas, axis=1)[0][:len(gamma), :][:, len(gamma):]

        A = np.zeros(C.shape)
        idx = np.argsort(np.mean(C, axis=1))
        for k1 in idx:
            k2 = np.argmax(C[k1, :])
            A[k1, k2] = 1
            C[k1, :] = -np.inf
            C[:, k2] = -np.inf

        alpha = np.dot(alpha, A)

        self.gamma_hat = np.dot(A.T, gamma)

        return alpha
