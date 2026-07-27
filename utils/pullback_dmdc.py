import numpy as np
import warnings
from eofs.standard import Eof
from sklearn.base import BaseEstimator
from typing import Optional, Dict


class PullbackDMDc(BaseEstimator):
    def __init__(self, 
                 truncation=20, 
                 lag=3, 
                 transition_time=1200):
        self.truncation = truncation
        self.lag = lag
        self.transition_time = transition_time

    def __getstate__(self):
        # Return only what you want to save
        state = self.__dict__.copy()
        # Remove unwanted attributes
        for k in ['_short_forcings','_long_forcings', '_G']:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        # Restore only the saved attributes
        self.__dict__.update(state)
        # Recreate anything that shouldn't be pickled

    def fit(self, data: np.array, short_forcings: np.array, long_forcings: np.array, precomputed_eofs: Optional[Dict] = None):

        self.run_eofs(data, precomputed_eofs)
 
        self._short_forcings = short_forcings
        self._long_forcings = long_forcings

        inputs = np.concatenate([self._pcs[:-self.lag,:], self._short_forcings[self.lag:,:]],axis=-1)
        targets = self._pcs[self.lag:,:]
        self._G = np.linalg.lstsq(inputs,targets,rcond=None)[0]
        # Extract and store the A (state) and B (forcing) blocks from G for later eigenvalue analysis
        self.A = self._G[:self.truncation,:].T
        self.B = self._G[self.truncation:,:].T
        
        #evolve the forced response
        t = self._long_forcings.shape[0]
        forced_response_pcs = np.zeros((t,self.truncation))
        for n in range(self.lag,t):
            forced_response_pcs[n,:]=self._G.T@np.concatenate([forced_response_pcs[n-self.lag,:],self._long_forcings[n,:]],axis=-1)

        self.fr_pcs = forced_response_pcs[self.transition_time:,:]

    def predict(self):
        return self.fr_pcs @ self.eofs + self.data_mean

    def run_eofs(self, data: np.array, precomputed_eofs: Optional[Dict] = None):
        if precomputed_eofs is None:
            self.data_mean = data.mean(axis = 0)
            data_centered = data - self.data_mean

            eofs_xr = Eof(data_centered, center=False, ddof=1)
            self.eofs = eofs_xr.eofs(neofs=self.truncation)
            self._pcs = eofs_xr.pcs(npcs=self.truncation)
        else:
            self.data_mean = precomputed_eofs['data_mean'].copy()
            self.eofs = precomputed_eofs['eofs'][:self.truncation,:].copy()
            self._pcs = precomputed_eofs['pcs'][:,:self.truncation].copy()

    def compute_modes(self):
        # Compute the eigenvalues and eigenvectors of the A matrix
        eigvals, eigvecs = np.linalg.eig(self.A)

        # Sort the eigenvalues and corresponding eigenvectors by magnitude
        idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals = eigvals[idx]
        self.W = eigvecs[:, idx]
        self.W_inv = np.linalg.inv(self.W)

        # Compute the DMD modes (spatial patterns)
        self.modes = self.eofs.T @ self.W
    
        # Compute the time dynamics (time series) for each mode
        self.full_time_series = (self.W_inv @ self._pcs.T).T
        self.forced_time_series = (self.W_inv @ self.fr_pcs.T).T # projection of pullback attractor, instead of its direct estimation here
        self.internal_time_series = self.full_time_series - self.forced_time_series


    @staticmethod
    def _is_conjugate_pair(eigvals: np.ndarray, k: int, n_modes: int) -> bool:
        if k + 1 >= n_modes:
            return False
        if np.isclose(np.imag(eigvals[k]), 0.0):
            return False
        return np.allclose(eigvals[k], np.conj(eigvals[k + 1]))

    @staticmethod
    def _pca_rotate_pair(W: np.ndarray, Z: np.ndarray, Z_f: np.ndarray):
        QW, RW = np.linalg.qr(W)
        QZ, RZ = np.linalg.qr(Z)
        U, _, _ = np.linalg.svd(RW @ RZ.T)
        W_pc = QW @ U
        P = RW.T @ U
        return W_pc, Z @ P, Z_f @ P

    def compute_rotated_modes(self, top_n: Optional[int] = None):
        """
        Return mode-wise spatial patterns and time series with conjugate pairs
        PCA-rotated into real-valued orthogonal components.

        Returns a dictionary with keys:
            - spatial_patterns: (n_space, n_modes)
            - forced_time_series: (n_time, n_modes)
            - internal_time_series: (n_time, n_modes)
            - labels: list[tuple[str, str]] with ("k", "real"),
              ("k,k+1", "complex"), or ("k,k+1", "complex-cutoff")
            - eigvals: (n_modes,)

        If top_n ends on the first half of a conjugate pair, the final slot is
        filled with the leading PCA rotation of that pair and a warning is
        emitted.
        """
        if not hasattr(self, 'eigvals') or not hasattr(self, 'modes'):
            self.compute_modes()

        n_total = self.eigvals.size
        n_modes = n_total if top_n is None else min(int(top_n), n_total)

        cutoff_pair = (
            n_modes > 0
            and n_modes < n_total
            and not np.isclose(np.imag(self.eigvals[n_modes - 1]), 0.0)
            and np.allclose(self.eigvals[n_modes - 1], np.conj(self.eigvals[n_modes]))
        )
        if cutoff_pair:
            warnings.warn(
                'top_n cuts off a complex-conjugate pair; the last returned '
                'slot will hold the leading PCA rotation of the pair.',
                RuntimeWarning,
                stacklevel=2,
            )

        spatial_patterns = np.zeros((self.modes.shape[0], n_modes), dtype=float)
        forced_ts = np.zeros((self.forced_time_series.shape[0], n_modes), dtype=float)
        internal_ts = np.zeros((self.internal_time_series.shape[0], n_modes), dtype=float)
        labels = [None] * n_modes

        k = 0
        while k < n_modes:
            is_cutoff_pair = (
                cutoff_pair
                and k == n_modes - 1
                and not np.isclose(np.imag(self.eigvals[k]), 0.0)
                and np.allclose(self.eigvals[k], np.conj(self.eigvals[k + 1]))
            )
            if is_cutoff_pair:
                W = np.stack(
                    [2.0 * np.real(self.modes[:, k]), -2.0 * np.imag(self.modes[:, k])],
                    axis=1,
                )
                Z = np.stack(
                    [
                        np.real(self.internal_time_series[:, k]),
                        np.imag(self.internal_time_series[:, k]),
                    ],
                    axis=1,
                )
                Z_f = np.stack(
                    [
                        np.real(self.forced_time_series[:, k]),
                        np.imag(self.forced_time_series[:, k]),
                    ],
                    axis=1,
                )
                W_pc, Z_pc, Z_f_pc = self._pca_rotate_pair(W=W, Z=Z, Z_f=Z_f)
                spatial_patterns[:, k] = W_pc[:, 0]
                internal_ts[:, k] = Z_pc[:, 0]
                forced_ts[:, k] = Z_f_pc[:, 0]
                labels[k] = (f'{k + 1},{k + 2}', 'complex-cutoff')
                k += 1
                continue

            if self._is_conjugate_pair(self.eigvals, k, n_modes):
                W = np.stack(
                    [2.0 * np.real(self.modes[:, k]), -2.0 * np.imag(self.modes[:, k])],
                    axis=1,
                )
                Z = np.stack(
                    [
                        np.real(self.internal_time_series[:, k]),
                        np.imag(self.internal_time_series[:, k]),
                    ],
                    axis=1,
                )
                Z_f = np.stack(
                    [
                        np.real(self.forced_time_series[:, k]),
                        np.imag(self.forced_time_series[:, k]),
                    ],
                    axis=1,
                )
                W_pc, Z_pc, Z_f_pc = self._pca_rotate_pair(W=W, Z=Z, Z_f=Z_f)
                spatial_patterns[:, k:k + 2] = W_pc
                internal_ts[:, k:k + 2] = Z_pc
                forced_ts[:, k:k + 2] = Z_f_pc
                pair_label = (f'{k + 1},{k + 2}', 'complex')
                labels[k] = pair_label
                labels[k + 1] = pair_label
                k += 2
            else:
                spatial_patterns[:, k] = np.real(self.modes[:, k])
                internal_ts[:, k] = np.real(self.internal_time_series[:, k])
                forced_ts[:, k] = np.real(self.forced_time_series[:, k])
                labels[k] = (f'{k + 1}', 'real')
                k += 1

        return {
            'spatial_patterns': spatial_patterns,
            'forced_time_series': forced_ts,
            'internal_time_series': internal_ts,
            'labels': labels,
            'eigvals': self.eigvals[:n_modes],
        }


