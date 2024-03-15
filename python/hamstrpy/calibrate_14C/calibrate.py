import numpy as np
from .log_prob import get_curve, log_prob_t
from .calibration_curves import intcal20, shcal20, marine20


curve_dict = {
    'IntCal20': intcal20,
    'SHCal20': shcal20,
    'Marine20': marine20,
}


def calibrate_14C_age(age, error, curve='IntCal20', func=log_prob_t):
    vals, curve = get_curve(
        age,
        error,
        func=func,
        calibration_curve=curve_dict[curve],
    )
    norm = (
        (vals[1:] - vals[:-1]) * 0.5 * (curve[1:] + curve[:-1])
    ).sum()
    curve /= norm
    mean = (
        (vals[1:] - vals[:-1])
        * 0.5
        * (curve[1:] * vals[1:] + curve[:-1] * vals[:-1])
    ).sum()
    secmom = (
        (vals[1:] - vals[:-1])
        * 0.5
        * (curve[1:] * vals[1:]**2 + curve[:-1] * vals[:-1]**2)
    ).sum()
    std = np.sqrt(secmom - mean**2) * (len(vals) - 1) / len(vals)
    cdf = np.cumsum(
        (vals[1:] - vals[:-1]) * 0.5 * (curve[1:] + curve[:-1])
    )
    med_ind = max(np.argwhere(cdf < 0.5).flatten())
    median = 0.5*(vals[1:] + vals[:-1])[med_ind]

    return mean, median, std


def calibrate_rc_data(rc_data, curve='IntCal20'):
    for ind, dat in rc_data.iterrows():
        try:
            _, median, std = calibrate_14C_age(
                dat['age'],
                dat['error'],
                curve=dat['curve'],
            )
        except KeyError:
            _, median, std = calibrate_14C_age(
                dat['age'],
                dat['error'],
                curve=curve,
            )
        rc_data.loc[ind, 't'] = median
        rc_data.loc[ind, 'dt'] = std
    return rc_data
