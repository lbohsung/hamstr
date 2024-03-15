import numpy as np

from .calibration_curves import intcal20


def log_prob_norm(t, age, error, calibration_curve=intcal20):
    # Normal distribution
    # XXX:not normalized
    mu = np.interp(
        t,
        1950 - calibration_curve['CAL BP'].values,
        calibration_curve['14C age'].values,
    )

    sig = np.interp(
        t,
        1950 - calibration_curve['CAL BP'].values,
        calibration_curve['Sigma 14C'].values,
    )

    sig = np.sqrt(error**2 + sig**2)
    df = mu - age
    return - 0.5 * (df / sig)**2 - np.log(sig) - 0.5 * np.log(2*np.pi)


def log_prob_t(t, age, error, a=3, b=4, calibration_curve=intcal20):
    # Generalized Student's t-distribution, as proposed by Christen and Perez
    # (2009)
    # XXX:not normalized
    mu = np.interp(
        t,
        1950 - calibration_curve['CAL BP'].values,
        calibration_curve['14C age'].values,
    )

    sig = np.interp(
        t,
        1950 - calibration_curve['CAL BP'].values,
        calibration_curve['Sigma 14C'].values,
    )

    sig = np.sqrt(error**2 + sig**2)
    df = mu - age
    return -(a+0.5) * np.log(b + 0.5 * (df / sig)**2)


def eval_calibration_curve(
    age,
    error,
    thresh=1e-3,
    func=log_prob_norm,
    calibration_curve=intcal20,
):
    _t = 1950 - calibration_curve['CAL BP'].values
    prob = np.exp(func(_t, age, error, calibration_curve=calibration_curve))
    prob /= np.sum(prob)
    inds = np.argwhere(thresh*prob.max() <= prob).flatten()

    return _t[min(inds):max(inds)], prob[min(inds):max(inds)]
