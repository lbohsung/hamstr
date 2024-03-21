import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import statsmodels
from statsmodels.formula.api import rlm
import statsmodels.api as sm


def make_stan_dat_hamstr(**kwargs):
    """ Make the data object required by the Stan sampler

    Parameters
    ----------
        **kwargs passed from the hamstr function

    Returns
    -------
    dict
        A dictionary containing the data and paramters to be passed as data to
        the Stan sampler
    """
    args = kwargs.copy()
    hamstr_control_default = {
        'scale_R': True,
        'nu': 6,
        'scale_shape': True,
        'smooth_s': False,
        'inflate_errors': False,
        'infl_sigma_sd': None,
        'infl_shape_shape': 1,
        'infl_shape_mean': 1,
    }

    for key, value in hamstr_control_default.items():
        try:
            args[key]
        except KeyError:
            args[key] = value

    # Calculate acc_mean_prior if not provided
    if args['acc_mean_prior'] is None:
        d = pd.DataFrame({
            'depth': args['depth'],
            'obs_age': args['obs_age']
        })
        model_result = rlm(
            'obs_age ~ depth',
            data=d,
            M=statsmodels.robust.norms.HuberT(),
        ).fit()
        # signif equivalent, see https://stackoverflow.com/a/56974893
        digits = 2 - np.int64(
            np.ceil(np.log10(np.abs(model_result.params['depth'])))
        )
        acc_mean = np.round(model_result.params['depth'], digits)
        acc_mean = 20 if acc_mean <= 0 else acc_mean
        args['acc_mean_prior'] = acc_mean

    # Sort depth and related arrays
    sorted_indices = np.argsort(args['depth'], kind='stable')
    for key in ['depth', 'obs_age', 'obs_err']:
        args[key] = np.array(args[key])[sorted_indices]

    # Handle model_bioturbation parameters
    if args.get('model_bioturbation', False):
        if 'L_prior_sigma' in args and args['L_prior_sigma'] is not None:
            print("L_prior_shape is being overridden by L_prior_sigma.")
        if not ('L_prior_sigma' in args or 'L_prior_shape' in args):
            raise ValueError(
                "One of either L_prior_sigma or L_prior_shape must be "
                "specified.\nSet either to 0 to impose a fixed mixing depth."
            )
        if not isinstance(args["n_ind"], (int, float, list)):
            raise ValueError(
                "n_ind must be either a single value or a vector the same "
                "length as obs_age"
            )
        if isinstance(args["n_ind"], (int, float)):
            args["n_ind"] = np.repeat(args["n_ind"], len(args["obs_age"]))
        if len(args["n_ind"]) > 1:
            args['n_ind'] = np.array(args["n_ind"])[sorted_indices]
        if 'L_prior_sigma' in args and args['L_prior_sigma'] is not None:
            if args["L_prior_sigma"] == 0:
                args['L_prior_shape'] = 0
            else:
                args['L_prior_shape'] = gamma_sigma_shape(
                    mean=args['L_prior_mean'], sigma=args['L_prior_sigma']
                )["shape"]
    else:
        args['n_ind'] = []

    # Set infl_sigma_sd if not provided
    if args['infl_sigma_sd'] is None:
        args['infl_sigma_sd'] = 10 * np.mean(args['obs_err'])

    # Handle top_depth and bottom_depth
    depth_keys = ['top_depth', 'bottom_depth']
    for key in depth_keys:
        if key not in args or args[key] is None:
            args[key] = args['depth'][0] if 'top' in key else args['depth'][-1]

    # Validate and calculate K_fine and K_factor
    if 'K_fine' not in args or args['K_fine'] is None:
        K_fine_1 = args['bottom_depth'] - args['top_depth']
        median_depth_diff = np.median(
            np.diff(np.sort(np.unique(args['depth'])))
        )
        K_fine_2 = np.round(16 * K_fine_1 / median_depth_diff)
        K_fine = min(K_fine_1, K_fine_2)
        args['K_fine'] = K_fine if K_fine <= 900 else 900

    if 'K_factor' not in args or args['K_factor'] is None:
        args['K_factor'] = get_K_factor(args['K_fine'])

    args['N'] = len(args['depth'])

    # Calculate breakpoints and indices
    brks = get_brks_half_offset(args['K_fine'], args['K_factor'])
    alpha_idx = get_indices(brks=brks)

    args['K_tot'] = np.sum(alpha_idx["nK"])
    args['K_fine'] = alpha_idx['nK'][-1]
    args['c'] = list(range(1, args['K_fine'] + 1))

    args['mem_alpha'] = args['mem_strength'] * args['mem_mean']
    args['mem_beta'] = args['mem_strength'] * (1 - args['mem_mean'])

    # Calculate depth section and smoothing indices
    args['delta_c'] = (
        (args['bottom_depth'] - args['top_depth']) / args['K_fine']
    )
    args['c_depth_bottom'] = (
        [args['delta_c'] * c for c in args['c']] + args['top_depth']
    )
    args['c_depth_top'] = np.concatenate(
        [[args['top_depth']], args['c_depth_bottom'][:args['K_fine'] - 1]]
    )

    args["modelled_depths"] = np.concatenate(
        [[args['c_depth_top'][0]], args['c_depth_bottom']]
    )

    args['which_c'] = [
        np.argmax(
            (args['c_depth_bottom'] < d) * (args['c_depth_bottom'] - d)
        ) + 1
        for d in args['depth']
    ]

    args.update(alpha_idx)
    args["n_lvls"] = len(args["nK"]) - 1

    for key in [
        'scale_shape',
        'model_bioturbation',
        'model_displacement',
        'smooth_s',
        'model_hiatus',
    ]:
        args[key] = int(args[key])

    if 'H_top' not in args or args['H_top'] is None:
        args['H_top'] = args['top_depth']
    if 'H_bottom' not in args or args['H_bottom'] is None:
        args['H_bottom'] = args['bottom_depth']

    if args['smooth_s'] == 1:
        args['smooth_i'] = get_smooth_i(args, args['L_prior_mean'])
        args['I'] = len(args['smooth_i'])
    else:
        args['smooth_i'] = np.array([1] * args['N']).reshape(1, -1)
        args['I'] = 1

    args.pop("L_prior_sigma")
    args.pop("brks")
    return args


def get_wts(a, b):
    """ Get weights for parent sections

    Parameters
    ----------
    a : array-like
        The parent breaks
    b : array-like
        The child breaks

    Returns
    -------
    array
        The weights
    """
    intvls = [b[i-1:i+1] for i in range(1, len(b))]
    gaps = [a[(a >= x[0]) & (a <= x[1])] for x in intvls]
    wts = []
    for i in range(len(intvls)):
        combined = np.unique(np.sort(np.concatenate([gaps[i], intvls[i]])))
        diffs = np.diff(combined)
        if len(diffs) == 0:
            diffs = [1]
        range_val = np.ptp(combined)
        wts_i = diffs / range_val if range_val != 0 else diffs
        wts.append(wts_i if len(wts_i) > 1 else np.repeat(wts_i, 2))
    wts = np.hstack([wt.reshape(-1, 1) for wt in wts])
    return wts


def get_indices(nK=None, brks=None):
    """ Get the index structure for the hamstr model from a set of breaks

    Parameters
    ----------
    nK : array-like
        List of number of breaks in each level
    brks : list of arrays
        A list of breakpoints in each level
    """
    if brks is None:
        lvl = np.concatenate(
            [np.repeat(i, n) for i, n in enumerate(nK, start=1)]
        )
        brks = [np.linspace(0, 1, num=x + 1) for x in nK]
    if nK is None:
        nK = [len(b) - 1 for b in brks]
        lvl = np.concatenate(
            [np.repeat(i, n) for i, n in enumerate(nK, start=1)]
        )
    parent = []
    for i in range(1, len(brks)):
        pa = np.digitize(brks[i][:-1], brks[i-1], right=False)
        pb = np.digitize(brks[i][1:], brks[i-1], right=True)
        parent.append(np.vstack([pa, pb]))
    cumNcol = np.cumsum([p.max() for p in parent[:-1]])
    for i, p in enumerate(parent[1:], start=1):
        parent[i] = p + cumNcol[i-1]
        wts = [
            get_wts(
                np.array(brks[i-1]),
                np.array(brks[i])
            ) for i in range(1, len(brks))
        ]
    parent = np.hstack(parent) if parent else np.array([[], []])
    multi_parent_adj = np.mean(np.abs(np.diff(parent, axis=0)) + 1)
    wts = np.hstack(wts) if wts else np.array([])
    if wts.size > 0:
        wts_normalized = np.apply_along_axis(
            lambda x: x / np.sum(x),
            0,
            wts.reshape(2, -1),
        )
        wts1, wts2 = wts_normalized[0, :], wts_normalized[1, :]
    else:
        wts1, wts2 = np.array([]), np.array([])

    return {
        'nK': nK,
        'alpha_idx': np.arange(1, np.sum(nK) + 1),
        'lvl': lvl,
        'brks': brks,
        'multi_parent_adj': multi_parent_adj,
        'parent1': parent[0, :].astype(int) if parent.size else np.array([]),
        'parent2': parent[1, :].astype(int) if parent.size else np.array([]),
        'wts1': wts1,
        'wts2': wts2,
    }


def get_brks_half_offset(K_fine, K_factor):
    """ Get the overlapping breaks structure """
    db_fine = 1 / K_fine
    db = db_fine
    brks = [np.arange(0, 1 + db, db)]
    n_br = len(brks[0])
    n_sec = n_br - 1
    newbrks = brks[0]
    while n_sec > 3:
        strt = np.min(brks[-1])
        # end = np.max(brks[-1])
        n_new = np.int64(np.ceil((n_sec + 1) / K_factor))
        l_new = n_new * K_factor
        l_old = n_sec
        d_new_old = l_new - l_old

        if d_new_old % 2 == 0:
            new_strt = strt - db * (d_new_old - 1) / 2
        else:
            new_strt = strt - db * (d_new_old) / 2
        newbrks = np.array(
            [new_strt + it * db * K_factor for it in range(n_new + 1)]
        )

        brks.append(newbrks)
        db = K_factor * db
        n_br = len(newbrks)
        n_sec = n_br - 1
    brks.append([newbrks[0], newbrks[-1]])
    brks = list(reversed(brks))
    # brks = [np.array(element) for element in brks]
    # brks = np.array([element.flatten() for element in brks])
    # brks = np.concatenate(brks)
    return brks


def get_K_factor(K_fine):
    """ Get the default K_factor

    Parameters
    ----------
    K_fine : int
        THe number of sections at the highest resolution
    """
    def bar(x, y):
        return abs(y - x**x)
    result = minimize_scalar(
        bar,
        bounds=(1, 10 + np.log10(K_fine)),
        args=(K_fine,),
        method='bounded',
    )
    return np.ceil(result.x)


def gamma_sigma_shape(mean=None, mode=None, sigma=None, shape=None):
    """ Convert between paremetrisations of the gamma distribution

    Parameters
    ----------
    mean : float
        The mean of the gamma distribution
    mode : float
        The mode of the gamma distribution
    sigma : float
        The standard deviation of the gamma distribution
    shape : float
        The hape parameter of the gamma distribution

    Returns
    -------
    dict
        A dictionary containing all parameters of the specified gamma
        distribution
    """
    if mean is None and mode is None:
        raise ValueError("One of either the mean or mode must be specified")
    if shape is None and sigma is None:
        raise ValueError("One of either the shape or sigma must be specified")
    if mean is not None and mode is not None:
        raise ValueError("Only one of the mean and mode can be specified")
    if shape is not None and sigma is not None:
        raise ValueError("Only one of the shape and sigma can be specified")
    if mean is None:
        if shape is not None:
            if shape <= 1:
                raise ValueError(
                    "Gamma cannot be defined by mode and shape if shape <= 1",
                )
            mean = (shape * mode) / (shape - 1)
        else:
            # from mode and sigma
            rate = (mode + np.sqrt(mode**2 + 4*sigma**2)) / (2 * sigma**2)
            shape = 1 + mode * rate
            if shape <= 1:
                raise ValueError(
                    "No solution for Gamma with this mode and sigma",
                )
            mean = (shape * mode) / (shape - 1)
    if sigma is None:
        # from mean and shape
        rate = shape / mean
        sigma = np.sqrt(shape / (rate**2))
    elif shape is None:
        # from mean and sigma
        rate = mean / sigma**2
        shape = mean**2 / sigma**2
    if mode is None:
        mode = (shape - 1) / rate
        if mode < 0:
            mode = 0

    return {
        'mean': mean,
        'mode': mode,
        'rate': rate,
        'shape': shape,
        'sigma': sigma,
    }


def get_smooth_i(d, w):
    """Get indices for smoothing accumulation rate when estimating L """
    w = w / d['delta_c']
    ri = np.arange(-np.floor(w / 2), np.floor(w / 2) + 1, dtype=int)
    mi = np.array([x + ri for x in d['which_c']])
    mi[mi <= 0] = np.abs(mi[mi <= 0]) + 1
    mi[mi > d['K_fine']] = 2 * d['K_fine'] - mi[mi > d['K_fine']] + 1
    if np.any(mi > d['K_fine']):
        raise ValueError("Acc rate smoothing index > K_fine")
    if np.any(mi < 1):
        raise ValueError("Acc rate smoothing index < 1")
    if not isinstance(mi, np.ndarray) or mi.ndim == 1:
        mi = np.array(mi, ndmin=2)
    return mi


def get_inits_hamstr(stan_dat):
    """ Create random initial values for the hamstr Stan model

    Parameters
    ----------
    stan_dat : dict
        A dictionary of data for the Stan hamstr model

    Returns
    -------
    dict
        A dictionary containing the initial values
    """
    X = sm.add_constant(stan_dat['depth'])
    model = sm.RLM(stan_dat['obs_age'], X).fit()
    sigma = model.bse[0]
    inits = {
        'R': np.random.uniform(0.1, 0.9, 1)[0],
        'alpha': np.abs(
            np.random.normal(
                stan_dat['acc_mean_prior'],
                stan_dat['acc_mean_prior'] / 3,
                stan_dat['K_tot'],
            )
        ),
        'age0': (
            model.predict([1, stan_dat['top_depth']])[0]
            + np.random.normal(0, sigma, 1)[0]
        )
    }
    if inits['age0'] < stan_dat['min_age']:
        inits['age0'] = stan_dat['min_age'] + np.abs(
            np.random.normal(0, 2, 1)[0]
        )
    if stan_dat.get('inflate_errors', 0) == 1:
        inits['infl_mean'] = np.abs(np.random.normal(0, 0.1, 1))
        inits['infl_shape'] = 1 + np.abs(np.random.normal(0, 0.1, 1))
        inits['infl'] = np.abs(np.random.normal(0, 0.1, stan_dat['N']))
    else:
        inits['infl_mean'] = np.array([])
        inits['infl_shape'] = np.array([])
        inits['infl'] = np.array([])
    return inits
