import arviz as az
import numpy as np
import pandas as pd

def get_posterior_ages(idata):
    modelled_depths = idata.observed_data['modelled_depths'].values
    posterior_ages = idata.posterior['c_ages'].values

    n_iters, n_depths = posterior_ages.shape[1], posterior_ages.shape[2]
    iter_idx, depth_idx = np.meshgrid(np.arange(n_iters), np.arange(n_depths), indexing='ij')
    df = pd.DataFrame({
        'iter': iter_idx.ravel() + 1,
        'depth': modelled_depths[depth_idx.ravel()],
        'age': posterior_ages[0, iter_idx.ravel(), depth_idx.ravel()]
    })
    df.sort_values(by=['iter', 'depth'], inplace=True)
    
    return df

