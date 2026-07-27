import numpy as np
import sys
import sklearn


import numpy as np, scipy
import sys
import sklearn



class TaylorDiagramEstimator:
    '''
        Classic TD

        X: [n_samples, ...]

        Y: [n_samples, ...]
    '''
    def __init__(self, centered=True, normalized=False):
        self.centered=centered
        self.normalized=normalized

    def inner_product(self, X: np.array, Y: np.array):
        #return (X*Y).mean()
        # print(X.shape,Y.shape,flush=True)
        N=len(X)
        if len(X.shape)>1:
            ip = np.einsum('ij,ij->',X,Y)/N
        else: 
            ip = (X*Y).sum()/N
        return ip

    def compute(self, X: np.array, Y: np.array):
        '''
            X and Y -- as defined in inner_product function
        '''
        if self.centered:
            X=X-X.mean(axis=0)
            Y=Y-Y.mean(axis=0)        
        cov_XX=self.inner_product(X,X)
        cov_YY=self.inner_product(Y,Y)
        cov_XY=self.inner_product(X,Y)
        return self.compute_TD_core(cov_XX,cov_XY,cov_YY)

    def compute_TD_core(self,cov_XX,cov_XY,cov_YY):
        '''
            X(ref)  and Y(prediction) -- as defined in inner_product function
        '''
        if np.abs(cov_XX)<=1e-10 and np.abs(cov_YY)<=1e-10 and np.abs(cov_XY)<=1e-10:
            cov_XX=1e-10
            cov_YY=1e-10
            cov_XY=1e-10
        # If all three are small, then it's 'comparing zero vs. zero' case. The 'normalized' diagram will return the reference point.
        
        std_X=np.sqrt(cov_XX)
        std_Y=np.sqrt(cov_YY)
        corr=cov_XY/(std_X*std_Y)
        mse=cov_XX+cov_YY-2.*cov_XY
        if mse>=-1e-10: mse=np.maximum(mse,0) # If numerical mse is slightly less then 0 (e.g. the case of constant time series).
        if mse>=0:
            rmse=np.sqrt(mse)
        else:
            print(f"Computed MSE ({mse}) is not semi-positive. Making RMSE=sqrt(abs(MSE)):", file=sys.stderr)
            rmse=np.sqrt(np.abs(mse))
        
        if self.normalized:
            std_Y/=std_X
            rmse/=std_X
            std_X=1.
        if np.abs(corr)<=1.01: corr=np.clip(corr,-1.,1.) # If numerical corr is slightly higher then 1.
        return {'std_X':std_X, 'std_Y':std_Y, 'corr':corr, 'rmse':rmse}

class PCA_TD_Estimator(TaylorDiagramEstimator):
    '''
        X: [n_samples, n_features]

        Y: list of [n_samples, truncation]

        V_Y: list of [truncation,n_features]
    '''
    def __init__(self, centered=True, normalized=False):
        super().__init__(centered=centered, normalized=normalized)

    def compute(self, X: np.array, P_Y: np.array, V_X: np.array=None, V_Y: np.array=None):
        P_X=X @ V_Y.T
        if self.centered:
            P_X=P_X-P_X.mean(axis=0)
            P_Y=P_Y-P_Y.mean(axis=0) 
        N=P_X.shape[0]
        cov_XX=np.einsum('ij,ij->',X, X)/N
        cov_YY=np.einsum('ij,ij->',P_Y, P_Y)/N
        cov_XY=np.einsum('ij,ij->',P_Y,P_X)/N
        return self.compute_TD_core(cov_XX,cov_XY,cov_YY) 

class Ensemble_PCA_TD_Estimator(TaylorDiagramEstimator):
    '''
        X: [n_samples, n_features]

        Y_list: list of [n_samples, truncation]

        V_Y_lust: list of [truncation,n_features] -- orthogonal matreices (no check is performed inside!)
    '''
    def __init__(self, centered=True, normalized=False):
        super().__init__(centered=centered, normalized=normalized)

    def compute(self, X: np.array, P_Y_list: list, V_Y_list: list):
        
        N=X.shape[0]
        if self.centered: X=X-X.mean(axis=0)
        cov_XX=np.einsum('ij,ij->',X, X)/N
        total_cov_XY=[]
        total_cov_YY=[]

        TD_list=[]
        for (P_Y,V_Y) in zip(P_Y_list,V_Y_list): 
            P_X=X @ V_Y.T
            if self.centered:
                P_Y=P_Y-P_Y.mean(axis=0) 
            cov_YY=np.einsum('ij,ij->',P_Y, P_Y)/N
            cov_XY=np.einsum('ij,ij->',P_X,P_Y)/N
            total_cov_YY.append(cov_YY)
            total_cov_XY.append(cov_XY)   

        TD_list=[self.compute_TD_core(cov_XX,cov_XY,cov_YY) for cov_YY, cov_XY in zip(total_cov_YY,total_cov_XY)]    
        total_TD=self.compute_TD_core(cov_XX,np.mean(total_cov_XY),np.mean(total_cov_YY))
        return TD_list, total_TD