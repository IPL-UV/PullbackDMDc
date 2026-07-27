import numpy as np
from eofs.standard import Eof
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from typing import Optional, Dict

class LR(BaseEstimator):
    def __init__(self, truncation=20):
        self.truncation = truncation

    def __getstate__(self):
        # Return only what you want to save
        state = self.__dict__.copy()
        # Remove unwanted attributes
        for k in ['_pcs','_forcings']:
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        # Restore only the saved attributes
        self.__dict__.update(state)
        # Recreate anything that shouldn't be pickled

    def fit(self, data: np.array, forcings: np.array, precomputed_eofs: Optional[Dict] = None):

        self._forcings = forcings

        self.run_eofs(data, precomputed_eofs)

        my_lr = LinearRegression()
        if len(self._forcings.shape)==1:
            inputs = self._forcings.reshape(-1, 1)
        elif len(self._forcings.shape)==2:
            inputs = self._forcings

        my_lr.fit(inputs, self._pcs)
        self.B = my_lr.coef_.copy()
        self.fr_pcs = my_lr.predict(inputs)

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