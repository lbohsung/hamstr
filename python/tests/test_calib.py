import pyreadr

from hamstrpy.calibrate_14C import calibrate_14C_age

rc_data = pyreadr.read_r(
    '../../data/MSB2K.rda',
)['MSB2K']

calibrate_14C_age(
    rc_data,
)

if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.stats import t

    from hamstrpy.calibrate_14C.log_prob import eval_calibration_curve

    rc_data['t (BP)'] = 1950 - rc_data['t']
    print(rc_data)

    inds = np.arange(40, step=np.floor(40/6))[0:6]

    fig, axs = plt.subplots(2, 3)

    for ind, at in enumerate(inds):
        ax_idx = np.unravel_index(
            ind,
            axs.shape,
        )
        vals, curve = eval_calibration_curve(
            rc_data.loc[at, 'age'],
            rc_data.loc[at, 'error'],
        )
        norm = (
            (vals[1:] - vals[:-1]) * 0.5 * (curve[1:] + curve[:-1])
        ).sum()
        curve /= norm

        axs[ax_idx].plot(
            (1950 - vals) / 1000,
            curve,
            color='C1',
        )
        axs[ax_idx].plot(
            (1950 - vals) / 1000,
            t.pdf(
                vals,
                df=6,
                loc=rc_data.loc[at, 't'],
                scale=rc_data.loc[at, 'dt'],
            ),
            color='C0',
        )

    plt.show()
