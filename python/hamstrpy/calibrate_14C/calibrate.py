import warnings

import numpy as np

import pandas as pd

from .log_prob import get_curve, log_prob_t
from .calibration_curves import intcal20, shcal20, marine20


curve_dict = {
    'IntCal20': intcal20,
    'SHCal20': shcal20,
    'Marine20': marine20,
}


def calibrate_14C_age(
    age_or_dataset,
    error=None,
    age_name='age',
    error_name='error',
    curve='IntCal20',
    func=log_prob_t,
):
    if isinstance(age_or_dataset, pd.Series):
        single_input = True
    elif np.isscalar(age_or_dataset):
        single_input = True
    else:
        single_input = len(age_or_dataset) == 1

    if single_input:
        if isinstance(age_or_dataset, (pd.Series, pd.DataFrame, dict)):
            age = age_or_dataset[age_name]
        else:
            age = age_or_dataset
        if error is None:
            error = age_or_dataset[error_name]
        vals, curve = get_curve(
            age,
            error,
            thresh=1e-5,
            func=func,
            calibration_curve=curve_dict[curve],
        )
        return summarize_empirical_pdf(vals, curve)
        # norm = (
        #     (vals[1:] - vals[:-1]) * 0.5 * (curve[1:] + curve[:-1])
        # ).sum()
        # curve /= norm
        # mean = (
        #     (vals[1:] - vals[:-1])
        #     * 0.5
        #     * (curve[1:] * vals[1:] + curve[:-1] * vals[:-1])
        # ).sum()
        # secmom = (
        #     (vals[1:] - vals[:-1])
        #     * 0.5
        #     * (curve[1:] * vals[1:]**2 + curve[:-1] * vals[:-1]**2)
        # ).sum()
        # std = np.sqrt(secmom - mean**2) * (len(vals) - 1) / len(vals)
        # cdf = np.cumsum(
        #     (vals[1:] - vals[:-1]) * 0.5 * (curve[1:] + curve[:-1])
        # )
        # med_ind = max(np.argwhere(cdf < 0.5).flatten())
        # median = vals[med_ind]

        # return mean, median, std

    else:
        for ind, dat in age_or_dataset.iterrows():
            try:
                _, median, std = calibrate_14C_age(
                    dat,
                    curve=dat['curve'],
                    func=func,
                )
            except KeyError:
                _, median, std = calibrate_14C_age(
                    dat,
                    curve=curve,
                    func=func,
                )

            age_or_dataset.loc[ind, 't'] = median
            age_or_dataset.loc[ind, 'dt'] = std

        return age_or_dataset


def summarize_empirical_pdf(x, p):
    inds = np.argsort(x)
    _p = p[inds]
    _x = x[inds]

    dx = x[1:] - x[:-1]

    if (max(abs(dx - np.mean(dx))) > np.median(dx) / 100):
        warnings.warn(
            "Empirical PDF has variable resolution - this is accounted for, "
            "but the results may be less reliable.",
            UserWarning,
        )

    dx = np.hstack(
        (
            [dx[0]],
            dx,
        ),
    )

    _p *= dx
    _p /= _p.sum()

    mean = np.sum(_x * _p)
    M = np.sum(_p > 0)
    std = np.sqrt(
        np.sum(
            _p * (_x - mean)**2
        )
        / ((M-1) / M * np.sum(_p))
    )
    cdf = np.cumsum(_p)
    med_ind = np.argmin(np.abs(cdf - 0.5)).flatten()
    median = _x[med_ind]

    return mean, median, std
