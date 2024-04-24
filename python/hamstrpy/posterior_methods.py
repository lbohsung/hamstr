from warnings import warn
from scipy.interpolate import interp1d
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


def get_interpolated_mean_and_std(idata):
    """
    Calculate the depth-to-time conversion functions based on posterior age
    distributions.

    This function computes the mean and standard deviation of the age
    distributions for modelled depths, and returns functions that interpolate
    these values across depths. The ages are adjusted to calendar years by
    subtracting them from 1950.

    Parameters
    ----------
    idata : arviz.InferenceData
        The sampling result returned by the hamstr function

    Returns
    -------
    d2t_mean : scipy.interpolate.interpolate.interp1d
        An interpolation function for the mean posterior age (in calendar
        years) as a function of modelled depth. This function can be called
        with an array of depths to obtain ages for depths of a given sediment
        record.
    d2t_std : scipy.interpolate.interpolate.interp1d
        An interpolation function for the standard deviation of the posterior
        ages (in calendar years) as a function of modelled depth. This function
        can be called with an array of depths to obtain ages uncertainties for
        depths of a given sediment record.

    Notes
    -----
    The interpolation kind is set to 'linear', and extrapolation is allowed for
    depths outside the range of the modelled depths. This behavior can be
    customized by modifying the 'kind' and 'fill_value' parameters of the
    interp1d functions within the code.
   """
    post_depths, post_ages = get_posterior_ages(idata)
    post_ages = 1950 - post_ages
    t_mean = post_ages.mean(axis=1)
    t_std = post_ages.std(axis=1)
    d2t_mean = interp1d(
        post_depths,
        t_mean,
        kind='linear',
        fill_value="extrapolate",
    )
    d2t_std = interp1d(
        post_depths,
        t_std,
        kind='linear',
        fill_value="extrapolate",
    )
    return d2t_mean, d2t_std


def interpolate_adm(idata):
    """
    Calculate the depth-to-time conversion functions based on posterior age
    distributions.

    This function linearly interpolates every posterior sample.
    The ages are adjusted to calendar years by subtracting them from 1950.

    Parameters
    ----------
    idata : arviz.InferenceData
        The sampling result returned by the hamstr function

    Returns
    -------
    interpolated_adm : scipy.interpolate.interpolate.interp1d
        An interpolation function for the all posterior samples (in calendar
        years) as a function of modelled depth. This function can be called
        with an array of depths to obtain posterior ages for depths of a given
        sediment record.

    Notes
    -----
    The interpolation kind is set to 'linear', and extrapolation is allowed for
    depths outside the range of the modelled depths. This behavior can be
    customized by modifying the 'kind' and 'fill_value' parameters of the
    interp1d functions within the code.
   """
    post_depths, post_ages = get_posterior_ages(idata)
    interpolated_adm = interp1d(
        post_depths,
        1950 - post_ages,
        kind='linear',
        fill_value="extrapolate",
        axis=0,
    )
    return interpolated_adm


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
