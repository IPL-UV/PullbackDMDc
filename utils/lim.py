import numpy as np
from eofs.standard import Eof
from sklearn.base import BaseEstimator
from typing import Optional, Dict

class LIM(BaseEstimator):
    def __init__(self, truncation=20, lag=3):
        self.truncation = truncation
        self.lag = lag

    def __getstate__(self):
        # Return only what you want to save
        state = self.__dict__.copy()
        # Remove unwanted attributes
        for k in ['_pcs','_UU','_VV','_u','_alpha','_D']:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        # Restore only the saved attributes
        self.__dict__.update(state)
        # Recreate anything that shouldn't be pickled

    def fit(self, data: np.array, precomputed_eofs: Optional[Dict] = None):

        self.run_eofs(data, precomputed_eofs)

        X0 = self._pcs[:,:-self.lag]
        Xtau = self._pcs[:,self.lag:]

        C0 = np.dot(X0, X0.T) / (X0.shape[1] - 1)
        Ctau = np.dot(Xtau, X0.T) / (X0.shape[1] - 1)

        # linear operator
        self.A = np.dot(Ctau,np.linalg.pinv(C0))
        
        D,U = np.linalg.eig(self.A)
        self._D = np.log(D)/self.lag
        V = np.linalg.inv(U).T

        # sort modes
        loc = np.argsort(-np.real(self._D))
        self._UU = U[:, loc]
        self._VV = V[:, loc]

        # identify the least damped mode
        self._u = self._UU[:, 0]
        v = self._VV[:, 0]
        self._alpha = np.dot(v, self._pcs)

        self.fr_pcs = np.real(np.outer(self._u, self._alpha)).T

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
        
        self._pcs=self._pcs.T

    def compute_modes(self):
        # Compute the eigenvalues and eigenvectors of the A matrix
        eigvals, eigvecs = np.linalg.eig(self.A)

        # Sort the eigenvalues and corresponding eigenvectors by magnitude
        idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals = eigvals[idx]
        self.W = eigvecs[:, idx]