import numpy as np
from eofs.standard import Eof
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt
from typing import Optional, Dict

class LIM_opt(BaseEstimator):
    def __init__(self, truncation=20, lag=3, optlag=60):
        self.truncation = truncation
        self.lag = lag
        self.optlag = optlag

    def __getstate__(self):
        # Return only what you want to save
        state = self.__dict__.copy()
        # Remove unwanted attributes
        for k in ['_pcs','__Psi','_Phi','_alphas','_limopts']:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        # Restore only the saved attributes
        self.__dict__.update(state)
        # Recreate anything that shouldn't be pickled

    def fit(self, data: np.array, precomputed_eofs: Optional[Dict] = None):

        self.run_eofs(data, precomputed_eofs)

        self.M = np.linalg.inv(self._pcs[:-self.lag,:].T @ self._pcs[:-self.lag,:]) @ self._pcs[:-self.lag,:].T @ self._pcs[self.lag:,:]
        
        Mnormed = self.M / np.linalg.norm(self.M)
        A = np.linalg.matrix_power(Mnormed, int(self.optlag/self.lag))

        self._Psi, _ ,self._Phit = np.linalg.svd(A, 0)

        self.fr_pcs = self._pcs @ self._Psi[:,0:1] @ self._Phit[0:1,:]

    def predict(self):
        return self.fr_pcs @ self.eofs + self.data_mean

    def run_eofs(self, data: np.array, precomputed_eofs: Optional[Dict] = None):
        if precomputed_eofs is None:
            self.data_mean = data.mean(axis = 0)
            data_centered = data - self.data_mean

            eofs_xr = Eof(data_centered, center=False, ddof=1)
            self.eofs = eofs_xr.eofs(neofs=self.truncation)
            self._pcs = eofs_xr.pcs(npcs=self.truncation)
            self.svals = np.sqrt(eofs_xr.eigenvalues(neigs=self.truncation))
        else:
            self.data_mean = precomputed_eofs['data_mean'].copy()
            self.eofs = precomputed_eofs['eofs'][:self.truncation,:].copy()
            self._pcs = precomputed_eofs['pcs'][:,:self.truncation].copy()
            self.svals = precomputed_eofs['svals'][:self.truncation].copy()

        if np.any(self.svals == 0):
            raise ValueError("Some singular values are zero, cannot scale eofs and pcs.")
        
        self._pcs = self._pcs @ np.diag(1/self.svals)
        self.eofs = np.diag(self.svals) @ self.eofs

