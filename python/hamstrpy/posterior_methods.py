from warnings import warn

import arviz as az


def get_posterior_ages(idata, thin=1):
    """ Extract the posterior ages from sampling result

    Parameters
    ----------
    idata : arviz.InferenceData
        The sampling result returned by the hamstr function
    thin : int, optional
        Thinning factor. This will slice the samples by ::thin

    Returns
    -------
    array of shape n
        The modelled dephts, useful for plotting etc.
    array of shape n x n_samples // thin
        The posterior age samples
    """
    modelled_depths = idata.observed_data['modelled_depths'].values
    posterior_ages = idata.posterior['c_ages'].values

    posterior_ages = posterior_ages.reshape(-1, modelled_depths.shape[0])
    posterior_ages = posterior_ages[::thin].T

    return modelled_depths, posterior_ages


def check_rhat(idata_or_summary, rhat, return_failed=False):
    """ Check if an rhat citerion is passed by all random variables

    Parameters
    ----------
    idata_or_summary : arviz.InferenceData or pandas.DataFrame
        Either a result returned by the hamstr function or the return value
        from a call to arviz.summary.
    rhat : float
        The rhat criterion to be checked.
    return_failed : bool, optional
        Whether to return the part of the summary that fails the criterion.

    Returns
    -------
    bool
        Whether all random variables pass the criterion.
    DataFrame
        A dataframe of failing random variables. Only returned if return_failed
        is True (default is False)
    """
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
