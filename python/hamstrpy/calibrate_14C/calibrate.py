import numpy as np

import pandas as pd

from .log_prob import eval_calibration_curve, log_prob_norm
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
    func=log_prob_norm,
):
    """ Calibrate radiocarbon data using reference curves.
    """
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
        x, pdf_at_x = eval_calibration_curve(
            age,
            error,
            thresh=1e-5,
            func=func,
            calibration_curve=curve_dict[curve],
        )
        return summarize_empirical_pdf(x, pdf_at_x)

    else:
        for ind, dat in age_or_dataset.iterrows():
            try:
                _, median, std = calibrate_14C_age(
                    dat,
                    age_name=age_name,
                    error_name=error_name,
                    curve=dat['curve'],
                    func=func,
                )
            except KeyError:
                _, median, std = calibrate_14C_age(
                    dat,
                    age_name=age_name,
                    error_name=error_name,
                    curve=curve,
                    func=func,
                )

            age_or_dataset.loc[ind, 't'] = 1950 - median
            age_or_dataset.loc[ind, 'dt'] = std

        return age_or_dataset


def summarize_empirical_pdf(x, pdf_at_x):
    """ Calculate mean, median and standard deviation of an empirical pdf by
    using trapezoid integration.
    """
    norm = (
        (x[1:] - x[:-1]) * 0.5 * (pdf_at_x[1:] + pdf_at_x[:-1])
    ).sum()
    pdf_at_x /= norm
    mean = (
        (x[1:] - x[:-1])
        * 0.5
        * (pdf_at_x[1:] * x[1:] + pdf_at_x[:-1] * x[:-1])
    ).sum()
    secmom = (
        (x[1:] - x[:-1])
        * 0.5
        * (pdf_at_x[1:] * x[1:]**2 + pdf_at_x[:-1] * x[:-1]**2)
    ).sum()
    std = np.sqrt(secmom - mean**2) * (len(x) - 1) / len(x)
    cdf = np.cumsum(
        (x[1:] - x[:-1]) * 0.5 * (pdf_at_x[1:] + pdf_at_x[:-1])
    )
    med_ind = max(np.argwhere(cdf < 0.5).flatten())
    median = x[med_ind]

    return mean, median, std
