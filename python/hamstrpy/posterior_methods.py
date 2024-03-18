from warnings import warn

import arviz as az


def get_posterior_ages(idata, thin=1):
    modelled_depths = idata.observed_data['modelled_depths'].values
    posterior_ages = idata.posterior['c_ages'].values

    posterior_ages = posterior_ages.reshape(-1, modelled_depths.shape[0])
    posterior_ages = posterior_ages[::thin].T

    return modelled_depths, posterior_ages


def check_rhat(idata_or_summary, rhat, return_failed=False):
    if isinstance(idata_or_summary, az.InferenceData):
        summary = az.summary(idata_or_summary)
    else:
        summary = idata_or_summary

    rhat_failed = summary.query(f"{rhat} < r_hat ")

    if 0 < len(rhat_failed):
        warn(
            'Not all random variables pass the R hat criterion',
            UserWarning,
        )
        print(rhat_failed)
        if return_failed:
            return False, rhat_failed
        return False
    else:
        return True
