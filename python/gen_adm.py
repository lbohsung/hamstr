import pandas as pd
import arviz as az
import stan_hamstr

data = pd.read_csv("./calib_rc_data.csv")
inference_data = stan_hamstr.hamstr(
    data.depth,
    data.age,
    data.error,
    # model_bioturbation=True,
    # n_ind = 5.0,
    # L_prior_sigma=2,
    stan_sampler_args = {"seed": 1}
)
az.to_netcdf(inference_data, "model_results.nc")